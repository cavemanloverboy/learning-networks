use std::f64::consts::PI;

use egui::{
    panel::Side,
    plot::{Bar, BarChart, Legend, Line, Plot},
    Color32,
};
use fabricator::{Factory, Network};
use tch::{nn, Tensor};
fn main() {
    // Load variables, factory from disk
    let device = tch::Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);

    // Define nn parameters. TODO: store in vs or elsewhere
    let generator_input: i64 = 1;
    let generator_width: i64 = 16;
    let generator_depth: i64 = 2;
    let generator_output: i64 = 1;
    let n_params: i64 = 2;
    let factory = Factory::new(
        &vs.root(),
        n_params,
        generator_input,
        generator_depth,
        generator_width,
        generator_output,
    );
    vs.load("generator.ot").expect("failed to load fabricator");

    start_up_demo(factory);
}

fn start_up_demo(factory: Factory) {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| Box::new(RNG::new(factory))),
    );
}

struct RNG {
    mu: f32,
    sigma: f32,
    factory: Factory,
    starting_up: bool,
}

impl eframe::App for RNG {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let update_plot = egui::CentralPanel::default()
            .show(ctx, |ui| {
                let mut update_plot = false;
                ui.heading("Learning Networks Demo");
                ui.label("Choose distribution parameters");
                update_plot = ui
                    .add(egui::Slider::new(&mut self.mu, -2.0..=2.0).text("mean"))
                    .changed();
                update_plot = ui
                    .add(egui::Slider::new(&mut self.sigma, 0.5..=2.0).text("std"))
                    .changed()
                    || update_plot;
                update_plot = ui.add(egui::Button::new("resample")).clicked() | update_plot;
                ui.label(format!("mean = {:.2}; std = {:.2}", self.mu, self.sigma));
                update_plot
            })
            .inner;
        egui::TopBottomPanel::bottom("bottom").show(ctx, |ui| {
            ui.heading("Learning Networks, Zamora et. al. 2022");
        });
        // if update_plot || self.starting_up {
        // self.starting_up = false;
        egui::SidePanel::new(Side::Right, "plot area").show(ctx, |ui| {
            let plot = Plot::new("Generator Demo")
                .legend(Legend::default())
                .include_x(-9.0)
                .include_x(9.0)
                .include_y(1.0)
                .view_aspect(1.0)
                .min_size(egui::Vec2 { x: 200.0, y: 200.0 });
            plot.show(ui, |plot_ui| {
                let (expected_line, generator_hist) = self.get_figures();
                plot_ui.bar_chart(generator_hist);
                plot_ui.line(expected_line);
            });
        });
        // }
    }
}

impl RNG {
    fn new(factory: Factory) -> Self {
        RNG {
            mu: 0.0,
            sigma: 1.0,
            factory,
            starting_up: true,
        }
    }
    fn get_figures(&self) -> (Line, BarChart) {
        // must be even, otherwise change bar positions
        const NUM_BARS: i64 = 32;
        const NUM_SAMPLES: i64 = 10_000;

        // Construct generator
        let parameters: [f32; 2] = [self.mu, self.sigma];
        let generator: Network = self.factory.manufacture_network(&parameters);

        // Get grid (sort shouldn't change results, just plot)
        let mut x: Vec<f32> = (0..NUM_SAMPLES).map(|_| rand::random()).collect();
        x.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let x_double: Vec<f64> = x.iter().map(|&f32_sample| f32_sample as f64).collect();
        let x_tensor: Tensor = Tensor::of_slice(&x).reshape(&[NUM_SAMPLES, 1]);

        // Propagate through generator
        let samples_tensor: Tensor = generator.forward(&x_tensor);
        let samples = Vec::<f32>::from(samples_tensor.contiguous().view(-1));

        // Create barcharts for samples
        let min = self.mu - 4.0 * self.sigma;
        let max = self.mu + 4.0 * self.sigma;
        let generator_bars = histogram(&samples, NUM_BARS, min, max);
        let generator_hist: BarChart = BarChart::new(generator_bars)
            .color(Color32::from_rgb(200, 100, 200))
            .name("generator");

        // Create line for expectation
        let expected_pairs: Vec<[f64; 2]> = x_double
            .into_iter()
            .map(|x| {
                [
                    x * 8.0 * self.sigma as f64 + self.mu as f64 - 4.0 * self.sigma as f64,
                    normal(
                        x * 8.0 * self.sigma as f64 + self.mu as f64 - 4.0 * self.sigma as f64,
                        self.mu as f64,
                        self.sigma as f64,
                    ),
                ]
            })
            .collect();
        let expected_line: Line = Line::new(expected_pairs)
            .color(Color32::from_rgb(100, 200, 100))
            .name("expected");

        // return line, chart
        (expected_line, generator_hist)
    }
}

fn normal(x: f64, mu: f64, sigma: f64) -> f64 {
    (2.0 * PI * sigma.powi(2)).recip().sqrt() * (-0.5 * ((x - mu) / sigma).powi(2)).exp()
}

fn histogram(values: &[f32], bins: i64, min: f32, max: f32) -> Vec<Bar> {
    assert!(
        max > min,
        "(histogram) max provided is not greater than min"
    );
    assert!(
        bins > 0,
        "(histogram) did not provide a positive number of bins"
    );

    // Calculate bar width
    let bar_width: f64 = (max - min) as f64 / bins as f64;

    // Calculate bar positions
    let bar_positions: Vec<f64> = (0..bins)
        .map(|i| min as f64 + (i as f64 + 0.5) * bar_width)
        .collect();

    // Initialize bars
    let mut bars: Vec<Bar> = bar_positions
        .iter()
        .map(|&p| Bar::new(p, 0.0).width(bar_width))
        .collect();

    // Increment bars
    let mut included = 0;
    for x in values {
        // Find which bin x belongs in
        let idx: usize = ((x - min) as f64 / bar_width) as usize;

        // If this bin exists, increment it
        if let Some(ref mut bar) = bars.get_mut(idx) {
            bar.value += 1.0;
            included += 1;
        }
    }
    // normalize
    for bar in &mut bars {
        bar.value /= included as f64 * bar_width;
    }
    bars
}
