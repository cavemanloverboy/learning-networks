use std::io::Write;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use plotly::{
    common::{Font, Mode, Title},
    layout::{Axis, AxisType},
    ImageFormat, Layout, NamedColor, Plot, Scatter,
};
use tch::nn::{self, OptimizerConfig};

use fabricator::{gen_batch, loss, Factory};

pub fn main() -> Result<()> {
    // Define nn parameters
    let emulator_input: i64 = 1;
    let emulator_width: i64 = 32;
    let emulator_depth: i64 = 2;
    let emulator_output: i64 = 1;
    let n_params: i64 = 3;

    // Training parameters
    const BATCHES_PER_EPOCH: usize = 1;
    const BATCH_SIZE: i64 = 128;
    const LOG_NUM_EPOCHS: u32 = 7;
    const TRAIN_EPOCHS: usize = 10_usize.pow(LOG_NUM_EPOCHS);

    // Create factory
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let factory = Factory::new(
        &vs.root(),
        n_params,
        emulator_input,
        emulator_depth,
        emulator_width,
        emulator_output,
    );

    // Where to save loss series
    const SAVE_PATH: &'static str = "loss";

    // Initialize optimizer
    let mut lr = 3e-4;
    let mut opt = nn::Adam::default().build(&vs, lr)?;

    let mut losses = Vec::with_capacity(TRAIN_EPOCHS);

    const RECORD_START: f64 = f64::INFINITY;
    let mut record = RECORD_START;

    let mut next_drop_epoch = 10;
    let mut next_report_epoch = 10_000;

    let pb = ProgressBar::new(TRAIN_EPOCHS as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}]{bar:20.cyan/blue}{pos:>5}/{len:5} {msg}")
            .unwrap(),
    );

    let mut loss_handles = Vec::with_capacity(100);
    for epoch in 1..=TRAIN_EPOCHS {
        let mut train_loss = 0f64;
        let mut samples = 0f64;
        for _ in 0..BATCHES_PER_EPOCH {
            // Generate batch, parameters
            let (x, params) = gen_batch(BATCH_SIZE);

            // Manufactor emulator and propagate input
            let emulator = factory.manufacture_network(&params);
            let output = emulator.forward(&x);

            // Calculate loss, and take optimizer step
            let loss = loss(&output, &x, params);
            opt.backward_step(&loss);

            // Update accumulators, counters
            train_loss += f64::from(&loss);
            samples += x.size()[0] as f64;
        }
        let epoch_loss = train_loss / samples;
        losses.push(epoch_loss);
        if epoch_loss < record {
            record = epoch_loss;
        }
        if epoch == next_drop_epoch {
            next_drop_epoch += 10;
            // lr *= 0.99997;
            // opt.set_lr(lr);
        }
        if epoch == next_report_epoch {
            next_report_epoch += 100_000;
            let loss_clone = losses.clone();
            loss_handles.push(std::thread::spawn(|| {
                plot_losses(loss_clone, LOG_NUM_EPOCHS);
            }));
        }

        // Update progress bar
        pb.inc(1);
        pb.set_message(format!(
            "L={epoch_loss:1.2e} best={record:1.2e} lr={lr:1.2e}"
        ));
    }

    // Wait for any outstanding handles
    for handle in loss_handles {
        handle.join().unwrap();
    }

    // Save emulator to disk
    vs.save("emulator.ot").expect("failed to save emulator");

    // Save and plot losses
    save_losses(&losses, SAVE_PATH);
    plot_losses(losses, LOG_NUM_EPOCHS);

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

    plot.save("emulator_losses.png", ImageFormat::PNG, 1024, 680, 1.0);
}
