use std::f32::consts::PI;

use fabricator::{Factory, Network};
use plotly::{
    common::{Font, Marker, Title},
    histogram::{Bins, HistNorm},
    layout::Axis,
    Bar, Histogram, ImageFormat, NamedColor, Rgba, Scatter,
};
use tch::{nn, Tensor};
fn main() {
    // Load variables, factory from disk
    let device = tch::Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);

    // Define nn parameters. TODO: store in vs or elsewhere
    let emulator_input: i64 = 1;
    let emulator_width: i64 = 64;
    let emulator_depth: i64 = 2;
    let emulator_output: i64 = 1;
    let n_params: i64 = 2;
    let factory = Factory::new(
        &vs.root(),
        n_params,
        emulator_input,
        emulator_depth,
        emulator_width,
        emulator_output,
    );
    vs.load("generator.ot").expect("failed to load fabricator");

    plot_densities(factory);
}

fn color(i: usize) -> NamedColor {
    match i {
        0 => NamedColor::Blue,
        1 => NamedColor::Crimson,
        2 => NamedColor::LimeGreen,
        _ => unreachable!("only 3 parabolas"),
    }
}

fn plot_densities(factory: Factory) {
    // Define normal distribution parameter sets
    const PARAMS: [[f32; 2]; 3] = [[-3.0, 0.5], [0.0, 0.7], [3.0, 0.6]];

    // Formatting
    const FONT_SIZE: usize = 20;

    // Grid/sample parameters
    const NUM_SIGMAS: f32 = 6.0;
    // must be even, otherwise change bar positions
    const NUM_BINS: i64 = 128;
    const NUM_SAMPLES: i64 = 100_000;
    const NUM_POINTS: i64 = 1_000;

    // Create plot
    let mut plot = plotly::Plot::new();

    // Change layout
    let layout = plotly::Layout::new()
        .legend(plotly::layout::Legend::new().font(Font::new().size(FONT_SIZE)))
        .y_axis(
            Axis::new()
                .title(
                    Title::from("Probability\tDensity")
                        .font(Font::new().size(FONT_SIZE))
                        .x(0.0),
                )
                .grid_color(NamedColor::DarkGray)
                .tick_font(Font::new().size(FONT_SIZE)),
        )
        .x_axis(
            Axis::new()
                .title(
                    Title::from("Output")
                        .font(Font::new().size(FONT_SIZE))
                        .x(0.0),
                )
                .tick_values(vec![-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
                .range(vec![-6.0, 6.0])
                .grid_color(NamedColor::DarkGray)
                .tick_font(Font::new().size(FONT_SIZE)),
        )
        .margin(
            plotly::layout::Margin::new()
                .left(80)
                .right(0)
                .top(80)
                .bottom(80),
        );
    plot.set_layout(layout);

    for (i, [mu, sigma]) in PARAMS.into_iter().enumerate() {
        // Generate generator for this parameter set
        let generator: Network = factory.manufacture_network(&[mu, sigma]);

        // Plot/sample parameters
        let min = mu - NUM_SIGMAS * sigma;
        let max = mu + NUM_SIGMAS * sigma;
        let size = (max - min) as f64 / NUM_BINS as f64;

        // Get grid (sort shouldn't change results, just plot)
        let mut x: Vec<f32> = (0..NUM_SAMPLES).map(|_| rand::random()).collect();
        x.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let x_tensor: Tensor = Tensor::of_slice(&x).reshape(&[NUM_SAMPLES, 1]);

        // Get actual density
        let expectation_x: Vec<f32> = (0..NUM_POINTS)
            .map(|i| min + (max - min) * (i as f32 + 0.5) / NUM_POINTS as f32)
            .collect();
        let expectation: Vec<f32> = expectation_x
            .iter()
            .map(|&x| normal(x, mu, sigma))
            .collect();
        let expectation_trace = Scatter::new(expectation_x, expectation)
            .line(plotly::common::Line::new().color(color(i)))
            .name(&format!("N{:?}", i));

        // Propagate through generator
        let samples_tensor: Tensor = generator.forward(&x_tensor);
        let samples = Vec::<f32>::from(samples_tensor.contiguous().view(-1));
        let (bins, hist) = histogram(&samples, NUM_BINS, min, max);

        // Create histogram for samples
        let generator_hist = Bar::new(bins, hist)
            .marker(
                Marker::new().color(color(i)).line(
                    plotly::common::Line::new()
                        .color(color(i))
                        .width(size * (1024.0 - 160.0) / 14.0),
                ),
            )
            .opacity(0.3)
            .show_legend(false);

        // Pick color, add traces
        plot.add_trace(expectation_trace);
        plot.add_trace(generator_hist);
    }

    plot.save("figure2", ImageFormat::PDF, 1024, 680, 1.0);
}

fn normal(x: f32, mu: f32, sigma: f32) -> f32 {
    (2.0 * PI * sigma.powi(2)).recip().sqrt() * (-0.5 * ((x - mu) / sigma).powi(2)).exp()
}

fn histogram(values: &[f32], bins: i64, min: f32, max: f32) -> (Vec<f32>, Vec<f32>) {
    assert!(
        max > min,
        "(histogram) max provided is not greater than min"
    );
    assert!(
        bins > 0,
        "(histogram) did not provide a positive number of bins"
    );

    // Calculate bar width
    let bar_width: f32 = (max - min) / bins as f32;

    // Calculate bar positions
    let mut bar_positions: Vec<f32> = (0..bins)
        .map(|i| min + (i as f32 + 0.5) * bar_width)
        .collect();

    // Initialize bars
    let mut bars: Vec<f32> = vec![0.0; bins as usize];

    // Increment bars
    let mut included = 0;
    for x in values {
        // Find which bin x belongs in
        let idx: usize = ((x - min) as f32 / bar_width) as usize;

        // If this bin exists, increment it
        if let Some(bar) = bars.get_mut(idx) {
            *bar += 1.0;
            included += 1;
        }
    }
    // normalize
    for bar in &mut bars {
        *bar /= included as f32 * bar_width;
    }

    // remove empty bars with a mutual drain filter
    let mut idx = 0;
    while idx < bars.len() {
        if bars[idx] == 0.0 {
            bars.remove(idx);
            bar_positions.remove(idx);
        } else {
            idx += 1;
        }
    }

    (bar_positions, bars)
}
