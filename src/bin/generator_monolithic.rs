use std::io::Write;

use anyhow::Result;

use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use plotly::{
    common::{Font, Mode, Title},
    layout::{Axis, AxisType},
    ImageFormat, Layout, NamedColor, Plot, Scatter,
};
use rand::{prelude::Distribution, thread_rng};
use statrs::distribution::{ContinuousCDF, Normal};
use tch::{
    nn::{self, linear, LinearConfig, Module, OptimizerConfig},
    Kind, Tensor,
};

/// The purpose of this is to train a factory-sized monolithic generator
fn main() -> Result<()> {
    // NN params
    const GENERATOR_INPUT: i64 = 3;
    const GENERATOR_OUTPUT: i64 = 1;
    const GENERATOR_WIDTH: i64 = 128;
    const GENERATOR_DEPTH: i64 = 5;

    // Training parameters
    const BATCHES_PER_EPOCH: usize = 1;
    const BATCH_SIZE: i64 = 32;
    const LOG_NUM_EPOCHS: u32 = 7;
    const TRAIN_EPOCHS: usize = 10_usize.pow(LOG_NUM_EPOCHS);

    // Create factory-sized generator
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let mut generator = nn::seq()
        .add(linear(
            &vs.root() / "fc1",
            GENERATOR_INPUT,
            GENERATOR_WIDTH,
            LinearConfig::default(),
        ))
        .add_fn(|xs| xs.relu());
    for l in 2..GENERATOR_DEPTH {
        generator = generator
            .add(linear(
                &vs.root() / format!("fc{l}"),
                GENERATOR_WIDTH,
                GENERATOR_WIDTH,
                LinearConfig::default(),
            ))
            .add_fn(|xs| xs.relu());
    }
    generator = generator.add(linear(
        &vs.root() / format!("fc{GENERATOR_DEPTH}"),
        GENERATOR_WIDTH,
        GENERATOR_OUTPUT,
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
            let (xms, z) = gen_sorted_batch::<BATCH_SIZE>();

            // Manufactor generator and propagate input
            let output = generator.forward(&xms);

            // Calculate loss, and take optimizer step
            let loss = loss(&output, &z);
            opt.backward_step(&loss);

            train_loss += f64::from(&loss);
            samples += xms.size()[0] as f64;
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
    let model_name: String = format!("{GENERATOR_WIDTH}_{GENERATOR_DEPTH}_gen");
    save_losses(&losses, &model_name);

    vs.save("monolithic_generator.ot").unwrap();

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
pub fn gen_sorted_batch<const SIZE: i64>() -> (Tensor, Tensor) {
    // // Generate random inputs
    let mut rng = thread_rng();
    let Result::<[Tensor; 2], _>::Ok([xms, z]) = Tensor::of_slice(
        &(0..SIZE)
            .map(|_| {
                // // mu in [-5, 5]
                let mu: f32 = rand::random::<f32>() * 10.0 - 5.0;
                // sigma in [0.5, 2]
                let sigma: f32 = rand::random::<f32>() * 1.5 + 0.5;
                let expected_generator = Normal::new(mu as f64, sigma as f64).unwrap();
                let z = expected_generator.sample(&mut rng);
                let x = expected_generator.cdf(z) as f32;
                [x, mu, sigma, z as f32]
            })
            .sorted_by(|a, b| a[0].partial_cmp(&b[0]).unwrap())
            .flatten()
            .collect::<Vec<f32>>(),
    )
    .reshape(&[SIZE, 4])
    .split_sizes(&[3, 1], 1)
    .try_into() else {
        panic!("failed to generate batch")
    };
    (xms, z)
}

pub fn loss(output: &Tensor, z: &Tensor) -> Tensor {
    (output - z).pow_tensor_scalar(2).sum(Kind::Double)
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
                .range(vec![1, log_total_num_epochs as i64])
                .type_(AxisType::Log)
                .grid_color(NamedColor::DarkGray),
        );
    plot.set_layout(layout);

    plot.save(
        "monolithic_gen_losses.png",
        ImageFormat::PNG,
        1024,
        680,
        1.0,
    );
}

#[test]
fn test_of_slice_split() {
    let tensor = Tensor::of_slice(&[1.0_f32, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    println!("{}", tensor.to_string(10).unwrap());
    let splits = tensor.split_sizes(&[1, 1], 1);
    println!(
        "{}, {:?}",
        splits.len(),
        splits
            .into_iter()
            .map(|t| t.to_string(10).unwrap())
            .collect::<Vec<String>>()
    );
}
