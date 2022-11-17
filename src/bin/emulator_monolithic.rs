use std::io::Write;

use anyhow::Result;

use indicatif::{ProgressBar, ProgressStyle};
use plotly::{
    common::{Font, Mode, Title},
    layout::{Axis, AxisType},
    ImageFormat, Layout, NamedColor, Plot, Scatter,
};
use tch::{
    nn::{self, linear, LinearConfig, Module, OptimizerConfig},
    Kind, Tensor,
};

/// The purpose of this is to train a factory-sized monolithic emulator
fn main() -> Result<()> {
    // NN params
    const EMULATOR_INPUT: i64 = 4;
    const EMULATOR_OUTPUT: i64 = 1;
    const EMULATOR_WIDTH: i64 = 128;
    const EMULATOR_DEPTH: i64 = 5;

    // Training parameters
    const BATCHES_PER_EPOCH: usize = 1;
    const BATCH_SIZE: i64 = 32;
    const LOG_NUM_EPOCHS: u32 = 7;
    const TRAIN_EPOCHS: usize = 10_usize.pow(LOG_NUM_EPOCHS);

    // Create factory-sized emulator
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let mut emulator = nn::seq()
        .add(linear(
            &vs.root() / "fc1",
            EMULATOR_INPUT,
            EMULATOR_WIDTH,
            LinearConfig::default(),
        ))
        .add_fn(|xs| xs.relu());
    for l in 2..EMULATOR_DEPTH {
        emulator = emulator
            .add(linear(
                &vs.root() / format!("fc{l}"),
                EMULATOR_WIDTH,
                EMULATOR_WIDTH,
                LinearConfig::default(),
            ))
            .add_fn(|xs| xs.relu());
    }
    emulator = emulator.add(linear(
        &vs.root() / format!("fc{EMULATOR_DEPTH}"),
        EMULATOR_WIDTH,
        EMULATOR_OUTPUT,
        LinearConfig::default(),
    ));
    println!(
        "num vars; {}",
        vs.variables_
            .lock()
            .unwrap()
            .trainable_variables
            .iter()
            .map(|var| var.tensor.size().into_iter().fold(1, |acc, x| acc * x))
            .sum::<i64>()
    );

    // Initialize optimizer
    let lr = 3e-4;
    let mut opt = nn::Adam::default().build(&vs, lr)?;

    const RECORD_START: f64 = f64::INFINITY;

    let mut record = RECORD_START;
    let mut next_drop_epoch = 10;
    let mut next_report_epoch = 100_000;

    let pb = ProgressBar::new(TRAIN_EPOCHS as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}; {eta_precise}]{bar:20.cyan/blue}{pos:>5}/{len:5} {msg}")
            .unwrap(),
    );

    let mut losses = Vec::with_capacity(TRAIN_EPOCHS);

    let mut loss_handles = Vec::with_capacity(TRAIN_EPOCHS / next_report_epoch);
    for epoch in 1..=TRAIN_EPOCHS {
        let mut train_loss = 0f64;
        let mut samples = 0f64;

        for _ in 0..BATCHES_PER_EPOCH {
            // Generate batch
            let xabc = gen_batch(BATCH_SIZE);

            // Manufactor emulator and propagate input
            let output = emulator.forward(&xabc);

            // Calculate loss, and take optimizer step
            let loss = loss(&output, &xabc);
            opt.backward_step(&loss);

            train_loss += f64::from(&loss);
            samples += xabc.size()[0] as f64;
        }
        // Calculate epoch loss
        let epoch_loss = train_loss / samples;
        losses.push(epoch_loss);

        // Update record if needed
        if epoch_loss < record {
            record = epoch_loss;
        }
        // exponential LR decay
        if epoch == next_drop_epoch {
            next_drop_epoch += 10;
            // lr *= 0.995;
            // opt.set_lr(lr);
        }
        if epoch == next_report_epoch {
            next_report_epoch += 100_000;
            let loss_clone = losses.clone();
            loss_handles.push(std::thread::spawn(|| {
                plot_losses(loss_clone, LOG_NUM_EPOCHS);
            }));
        }

        // Increment epoch counter and update pb message
        pb.inc(1);
        pb.set_message(format!(
            "L={epoch_loss:1.2e} best={record:1.2e} lr={lr:1.2e}"
        ));
    }

    // Finish training this model, save losses
    pb.finish();
    let model_name: String = format!("{EMULATOR_WIDTH}_{EMULATOR_DEPTH}_em");
    save_losses(&losses, &model_name);

    vs.save("monolithic.ot").unwrap();

    Ok(())
}

pub fn save_losses(losses: &Vec<f64>, path: &str) {
    let mut file = std::fs::File::create(path).unwrap();
    let loss_bytes = losses
        .into_iter()
        .map(|loss| loss.to_le_bytes())
        .flatten()
        .collect::<Vec<u8>>();
    file.write(&loss_bytes)
        .expect("failed to write losses to disk");
}

pub fn gen_batch(size: i64) -> Tensor {
    // // Generate random inputs
    let xabc: Vec<f32> = (0..4 * size)
        .map(|_| rand::random::<f32>() * 2.0 - 1.0)
        .collect();

    Tensor::of_slice(&xabc).reshape(&[size, 4])
}

pub fn loss(output: &Tensor, xabc: &Tensor) -> Tensor {
    let Result::<[Tensor; 4], _>::Ok([x, a, b, c]) = xabc.split_sizes(&[1, 1, 1, 1], 1).try_into() else { panic!() };
    let expected = c + (b * &x) + (a * &x * x);
    (output - expected).pow_tensor_scalar(2).sum(Kind::Double)
}

pub fn plot_losses(losses: Vec<f64>, log_total_num_epochs: u32) {
    // x-axis is just epochs
    let epochs: Vec<usize> = (1..=losses.len()).collect();

    // Create loss trace
    let loss_trace = Scatter::new(epochs, losses).mode(Mode::Lines).name("loss");

    // Create plot and add loss trace
    let mut plot = Plot::new();
    plot.add_trace(loss_trace);

    const FONT_SIZE: usize = 16;

    // Change layout
    let layout = Layout::new()
        .y_axis(
            Axis::new()
                .title(Title::from("Loss").font(Font::new().size(FONT_SIZE)).x(0.0))
                .type_(AxisType::Log)
                // .tick_values(
                //     vec![1.0, 5.0, 10.0, 20.0, 30.0, 40.0]
                //         .into_iter()
                //         .filter(|&x| x > MIN_R && x < MAX_R)
                //         .collect(),
                // )
                .grid_color(NamedColor::DarkGray),
        )
        .x_axis(
            Axis::new()
                .title(
                    Title::from("Epoch")
                        .font(Font::new().size(FONT_SIZE))
                        .x(0.0),
                )
                .range(vec![1, 6])
                .type_(AxisType::Log)
                .grid_color(NamedColor::DarkGray),
        );
    plot.set_layout(layout);

    plot.save("monolithic_losses.png", ImageFormat::PNG, 1024, 680, 1.0);
}
