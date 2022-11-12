use std::io::Write;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use plotly::{
    common::{Font, Mode, Title},
    layout::{Axis, AxisType},
    ImageFormat, Layout, NamedColor, Plot, Scatter,
};
use rand::{prelude::Distribution, thread_rng};
use statrs::distribution::{ContinuousCDF, Normal};
use tch::{
    nn::{self, OptimizerConfig},
    Kind, Tensor,
};

use fabricator::Factory;

pub fn main() -> Result<()> {
    // Define nn parameters
    let generator_input: i64 = 1;
    let generator_width: i64 = 64;
    let generator_depth: i64 = 2;
    let generator_output: i64 = 1;
    let n_params: i64 = 2;

    // Training parameters
    const BATCHES_PER_EPOCH: usize = 1;
    const BATCH_SIZE: i64 = 512;
    const TRAIN_EPOCHS: usize = 100_000;

    // Create factory
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let factory = Factory::new(
        &vs.root(),
        n_params,
        generator_input,
        generator_depth,
        generator_width,
        generator_output,
    );

    // Where to save loss series
    const SAVE_PATH: &'static str = "generator_loss";

    // Initialize optimizer
    let mut lr = 3e-4;
    let mut opt = nn::Adam::default().build(&vs, lr)?;

    let mut losses = Vec::with_capacity(TRAIN_EPOCHS);

    const RECORD_START: f64 = f64::INFINITY;
    let mut record = RECORD_START;

    let mut next_drop_epoch = 10;

    let pb = ProgressBar::new(TRAIN_EPOCHS as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}]{bar:20.cyan/blue}{pos:>5}/{len:5} {msg}")
            .unwrap(),
    );

    for epoch in 1..=TRAIN_EPOCHS {
        let mut train_loss = 0f64;
        let mut samples = 0f64;
        for _ in 0..BATCHES_PER_EPOCH {
            // Generate batch, parameters
            let (x, z, params) = gen_sorted_batch(BATCH_SIZE);

            // Manufactor generator and propagate input
            let generator = factory.manufacture_network(&params);
            let output = generator.forward(&x);

            // Calculate loss, and take optimizer step
            let loss = mse_loss(&output, &z); // skl_loss(&output, params) + 10.0_f64 * mt_loss(&output) + grad_loss(&output, &x);
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
            lr *= 0.99997;
            opt.set_lr(lr);
        }

        // Update progress bar
        pb.inc(1);
        pb.set_message(format!(
            "L={epoch_loss:1.2e} best={record:1.2e} lr={lr:1.2e}"
        ));
    }

    // Save generator to disk
    vs.save("generator.ot").expect("failed to save generator");

    save_losses(&losses, SAVE_PATH);
    plot_losses(losses);

    pb.finish();

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

pub fn plot_losses(losses: Vec<f64>) {
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
                .type_(AxisType::Log)
                .grid_color(NamedColor::DarkGray),
        );
    plot.set_layout(layout);

    plot.save("generator_losses.png", ImageFormat::PNG, 1024, 680, 1.0);
}

// KL divergence
fn skl_loss(z: &Tensor, [mu, sigma]: [f32; 2]) -> Tensor {
    // Calculate the mean and variance of the sample
    let sample_mu = z.mean_dim([0].as_ref(), false, Kind::Float);
    let sample_sigma = z.std_dim([0].as_ref(), true, false);

    // Calculate kl loss terms
    let term1: Tensor = (sigma / &sample_sigma).log();
    let term2: Tensor = (2.0 * sigma.powi(2)).recip()
        * (sample_sigma.pow_tensor_scalar(2) + (mu - &sample_mu).pow_tensor_scalar(2));
    let term1_2: Tensor = (sigma.recip() * &sample_sigma).log();
    let term2_2: Tensor = 1.0 / (2.0_f32 * sample_sigma.pow_tensor_scalar(2))
        * (sigma.powi(2) + (mu - sample_mu).pow_tensor_scalar(2));
    const TERM3: f64 = -0.5;

    (term1 + term2 + term1_2 + term2_2).sum(Kind::Double) + Tensor::from(2.0 * TERM3)
}

// monotonicity loss, expects z to be monotonic, so x must be monotonic
fn mt_loss(z: &Tensor) -> Tensor {
    -z.diff::<Tensor>(
        1,
        0,
        Some(Tensor::of_slice::<f32>(&[]).reshape(&[-1, 1])),
        Some(Tensor::of_slice::<f32>(&[]).reshape(&[-1, 1])),
    )
    .clamp_max(0.0_f64)
    .sum(Kind::Double)
}

fn grad_loss(z: &Tensor, x: &Tensor) -> Tensor {
    // let grad: Tensor = z.f_grad().unwrap();
    // let grad_type = grad.f_kind();
    // println!("grad type is {:?}", grad_type);
    // z.grad()
    //     .set_requires_grad(true)
    //     .pow_tensor_scalar(2)
    //     .clamp_min(1e3)
    //     .sum(Kind::Double)

    // let zgrad = Tensor::run_backward(
    //     &[z.set_requires_grad(true)],
    //     &[x.set_requires_grad(true)],
    //     true,
    //     true,
    // );
    // zgrad[0]
    //     .set_requires_grad(true)
    //     .pow_tensor_scalar(2)
    //     .clamp_min(1e3)
    //     .sum(Kind::Double)
    //     .set_requires_grad(true)

    // makeshift finite diff
    let dz_dx = z.diff::<Tensor>(
        1,
        0,
        Some(Tensor::of_slice::<f32>(&[]).reshape(&[-1, 1])),
        Some(Tensor::of_slice::<f32>(&[]).reshape(&[-1, 1])),
    ) / x
        .diff::<Tensor>(
            1,
            0,
            Some(Tensor::of_slice::<f32>(&[]).reshape(&[-1, 1])),
            Some(Tensor::of_slice::<f32>(&[]).reshape(&[-1, 1])),
        )
        .clamp_min(1e-10);
    (dz_dx.abs() - 1e1).clamp_min(0.0).sum(Kind::Double)
}

// pub fn gen_sorted_batch(size: i64) -> (Tensor, [f32; 2]) {
//     // Generate random parameters
//     // mu in [-2, 2]
//     let mu: f32 = rand::random::<f32>() * 4.0 - 2.0;
//     // sigma in [0.5, 2]
//     let sigma: f32 = rand::random::<f32>() * 1.5 + 0.5;

//     // Generate random inputs
//     let mut x: Vec<f32> = (0..size).map(|_| rand::random::<f32>()).collect();
//     x.sort_by(|a, b| a.partial_cmp(&b).unwrap());
//     let x = Tensor::of_slice(&x).reshape(&[size, 1]);

//     (x, [mu, sigma])
// }

pub fn gen_sorted_batch(size: i64) -> (Tensor, Tensor, [f32; 2]) {
    // Generate random parameters
    // mu in [-5, 5]
    let mu: f32 = rand::random::<f32>() * 10.0 - 5.0;
    // sigma in [0.5, 2]
    let sigma: f32 = rand::random::<f32>() * 1.5 + 0.5;

    // Generate random inputs
    let mut rng = thread_rng();
    let expected_generator = Normal::new(mu as f64, sigma as f64).unwrap();
    let z: Vec<f32> = (0..size)
        .map(|_| expected_generator.sample(&mut rng) as f32)
        .collect();
    let x: Vec<f32> = z
        .iter()
        .map(|&z| expected_generator.cdf(z as f64) as f32)
        .collect();
    let x = Tensor::of_slice(&x).reshape(&[size, 1]);
    let z = Tensor::of_slice(&z).reshape(&[size, 1]);

    (x, z, [mu, sigma])
}

fn mse_loss(output: &Tensor, z: &Tensor) -> Tensor {
    (output - z).pow_tensor_scalar(2.0).sum(Kind::Double)
}
