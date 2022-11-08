use std::io::Write;

use anyhow::Result;

use fabricator::{gen_batch, loss, Factory};
use indicatif::{ProgressBar, ProgressStyle};
use tch::nn::{self, OptimizerConfig};

/// The purpose of this suite is to find an appropriately small
/// network to serve as the emulator for the test function
/// f(x | a, b, c) = ax^2 + bx + c
fn main() -> Result<()> {
    // NN params
    const EMULATOR_INPUT: i64 = 1;
    const EMULATOR_OUTPUT: i64 = 1;
    const WIDTHS: &[i64] = &[4];
    const DEPTHS: &[i64] = &[0, 1, 2];

    // Training parameters
    const BATCHES_PER_EPOCH: usize = 1;
    const BATCH_SIZE: i64 = 128;
    const TRAIN_EPOCHS: usize = 100_000;
    const TEST_EPOCHS: usize = 1000;

    // Model params
    const PARAMS: [f32; 3] = [1.0, 1.0, 1.0];
    const NUM_PARAMS: i64 = 3;

    // Vecor which keeps records across all trials
    let mut test_losses = Vec::with_capacity(WIDTHS.len() * DEPTHS.len());

    for &emulator_width in WIDTHS {
        'depth: for &emulator_depth in DEPTHS {
            // already have this one. temp
            if [emulator_width, emulator_depth] == [8, 2] {
                continue 'depth;
            }
            // Create factory for this configuration
            let device = tch::Device::cuda_if_available();
            let vs = nn::VarStore::new(device);
            let factory = Factory::new(
                &vs.root(),
                NUM_PARAMS,
                EMULATOR_INPUT,
                emulator_depth,
                emulator_width,
                EMULATOR_OUTPUT,
            );

            // Initialize optimizer
            let lr = 3e-4;
            let mut opt = nn::Adam::default().build(&vs, lr)?;

            const RECORD_START: f64 = f64::INFINITY;

            let mut record = RECORD_START;
            let mut next_drop_epoch = 10;

            let pb = ProgressBar::new(TRAIN_EPOCHS as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}]{bar:20.cyan/blue}{pos:>5}/{len:5} {msg}")
                    .unwrap(),
            );

            let mut losses = Vec::with_capacity(TRAIN_EPOCHS);

            for epoch in 1..=TRAIN_EPOCHS {
                let mut train_loss = 0f64;
                let mut samples = 0f64;

                for _ in 0..BATCHES_PER_EPOCH {
                    // Generate batch, parameters
                    let (x, _) = gen_batch(BATCH_SIZE);
                    // if double {
                    //     gen_batch_double(BATCH_SIZE)
                    // } else {
                    // };

                    // Manufactor emulator and propagate input
                    let emulator = factory.manufacture_network(&PARAMS);
                    let output = emulator.forward(&x);

                    // Calculate loss, and take optimizer step
                    let loss = loss(&output, &x, PARAMS);
                    opt.backward_step(&loss);

                    train_loss += f64::from(&loss);
                    samples += x.size()[0] as f64;
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

                // Increment epoch counter and update pb message
                pb.inc(1);
                pb.set_message(format!(
                    "L={epoch_loss:1.2e} best={record:1.2e} lr={lr:1.2e}"
                ));
            }

            // Finish training this model, save losses
            pb.finish();
            let model_name: String = format!("{emulator_width}_{emulator_depth}");
            save_losses(&losses, &model_name);

            // Initialize test loss and denominator
            let mut test_loss: f64 = 0.0;
            let mut samples: f64 = 0.0;

            // Get a test loss, just need one emulator
            // Manufactor emulator
            let emulator = factory.manufacture_network(&PARAMS);
            for _ in 1..=TEST_EPOCHS {
                for _ in 0..BATCHES_PER_EPOCH {
                    // Generate batch, parameters
                    let (x, _) = gen_batch(BATCH_SIZE);

                    let output = emulator.forward(&x);
                    // Calculate loss, and take optimizer step
                    let loss = loss(&output, &x, PARAMS);

                    // Add to loss and denominator
                    test_loss += f64::from(&loss);
                    samples += x.size()[0] as f64;
                }
            }
            // Average
            test_loss /= samples;

            // Append record to vector
            test_losses.push(test_loss);
        }
    }

    // Print results
    let mut idx = 0;
    for emulator_width in WIDTHS {
        for emulator_depth in DEPTHS {
            println!(
                "{emulator_width:5} {emulator_depth:3}: {:1.3e}",
                test_losses[idx]
            );
            idx += 1;
        }
    }

    // save_losses(&test_losses, "test_losses");
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
