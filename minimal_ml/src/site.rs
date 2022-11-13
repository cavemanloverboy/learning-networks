use std::f64::consts::PI;

use super::{
    construct_emulator_factory, construct_generator_factory, utils::construct_network, Config,
    FullyConnected,
};
use egui::{
    panel::Side,
    plot::{Bar, BarChart, Legend, Line, LineStyle, Plot, PlotPoints},
    Color32, FontFamily, FontId, RichText, ScrollArea,
};

const EMULATOR_CONFIG: Config = Config {
    width: 16,
    hidden_layers: 3,
    in_dims: 1,
    out_dims: 1,
};

const GENERATOR_CONFIG: Config = Config {
    width: 64,
    hidden_layers: 3,
    in_dims: 1,
    out_dims: 1,
};
// must be even, otherwise change bar positions
const NUM_BARS: i64 = 32;
const NUM_SAMPLES: i64 = 1_000;

const POST: &'static str = include_str!("post.txt");

#[cfg(not(target_arch = "wasm32"))]
pub fn start_up_demo() {
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| Box::new(LearningNetworks::new())),
    );
}

#[cfg(target_arch = "wasm32")]
pub fn start_up_demo() {
    let options = eframe::WebOptions::default();

    println!("starting web");
    eframe::start_web(
        "learningnetworks",
        options,
        Box::new(|_cc| Box::new(LearningNetworks::new())),
    )
    .expect("failed to start eframe");
}

pub struct LearningNetworks {
    parabola: ParabolaEmulator,
    rng: RandomNumberGenerator,
}

struct ParabolaEmulator {
    a: f32,
    b: f32,
    c: f32,
    emulator_factory: FullyConnected,
    current_emulator: FullyConnected,
    // plot_bytes: Vec<u8>,
}
struct RandomNumberGenerator {
    mu: f32,
    sigma: f32,
    generator_factory: FullyConnected,
    current_generator: FullyConnected,
    current_samples: (Vec<f64>, Vec<f32>),
}

impl eframe::App for LearningNetworks {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            
            let total_width = ui.available_width();
            let total_height = ui.available_height();
            let font_scale = (total_width.min(1.5*total_height) / 1000.0).max(1.0);

                        
            ui.columns(2, |columns| {
                // Column 1: Text
                let (emulator_changed, generator_changed, resample): (bool, bool, bool) =
                    ScrollArea::vertical()
                        .id_source("Text")
                        .show(&mut columns[0], |ui| {
                            // Title of page
                            ui.heading(
                                RichText::new("Learning Networks")
                                    .font(FontId::proportional(20.0 * font_scale))
                                    .color(Color32::WHITE),
                            );

                            // Body
                            ui.label(
                                RichText::new(POST)
                                    .font(FontId::proportional(16.0 * font_scale))
                                    .color(Color32::WHITE),
                            );

                            // Parabola Sliders
                            ui.label(
                                RichText::new("\nChoose parabola parameters")
                                .font(FontId::proportional(16.0 * font_scale))
                                .color(Color32::WHITE)
                            );
                            let a = ui
                                .add(egui::Slider::new(&mut self.parabola.a, -1.0..=1.0).text("a"));
                            let b = ui
                                .add(egui::Slider::new(&mut self.parabola.b, -1.0..=1.0).text("b"));
                            let c = ui
                                .add(egui::Slider::new(&mut self.parabola.c, -1.0..=1.0).text("c"));

                            // RNG sliders
                            ui.label(
                                RichText::new("\nChoose rng parameters")
                                .font(FontId::proportional(16.0 * font_scale))
                                .color(Color32::WHITE)
                            );
                            let mu = ui
                                .add(egui::Slider::new(&mut self.rng.mu, -5.0..=5.0).text("avg"));
                            let sigma = ui
                                .add(egui::Slider::new(&mut self.rng.sigma, 0.5..=2.0).text("std"));
                            let resample = ui.button(
                                RichText::new("Resample")
                                .font(FontId::proportional(16.0 * font_scale))
                            ).clicked();

                            ui.label(
                                RichText::new(
                                    "\n\n\nLearning Networks: Ultralight Emulators, Generators, and Beyond (Zamora et. al. 2022)"
                                )
                                .font(FontId::proportional(16.0 * font_scale))
                            );
                            

                            let emulator_change = a.changed() || b.changed() || c.changed();
                            let generator_change = mu.changed() || sigma.changed();
                            (emulator_change, generator_change, resample)
                        })
                        .inner;

                // Column 2: Demo area
                egui::containers::Frame::default().show(&mut columns[1], |ui| {
                    //     .frame(columns[1])
                    // columns[1]
                    // egui::SidePanel::new(Side::Right, "Demos")
                    // .min_width(500.0)
                    // .show(ctx, |ui| {
                    // Get panel dimensions
                    let height = ui.available_height();
                    let width = ui.available_width();
                    let half_height = height / 2.0;

                    // On top half put parabola
                    let plot = Plot::new("parabola")
                        .legend(Legend::default())
                        .include_y(-2.1 + self.parabola.c)
                        .include_y(2.1 + self.parabola.c)
                        .width(width)
                        .height(half_height);
                    plot.show(ui, |plot_ui| {
                        let [expected_line, emulator_line] = self.get_lines(emulator_changed);
                        plot_ui.line(expected_line);
                        plot_ui.line(emulator_line);
                    });

                    let plot = Plot::new("Generator Demo")
                        .legend(Legend::default())
                        .include_x(-9.0)
                        .include_x(9.0)
                        .include_y(1.0)
                        .width(width)
                        .height(half_height);
                    plot.show(ui, |plot_ui| {
                        let (expected_line, generator_hist) =
                            self.get_figures(generator_changed, resample);
                        plot_ui.bar_chart(generator_hist);
                        plot_ui.line(expected_line);
                    });
                });
            });
        });
    }
}

impl LearningNetworks {
    pub fn new() -> Self {
        // Load emulator and generator
        let emulator_factory = construct_emulator_factory(&EMULATOR_CONFIG);
        let generator_factory = construct_generator_factory(&GENERATOR_CONFIG);

        // Construct default networks
        let default_emulator_wb = emulator_factory.forward(&[1.0, 0.0, 0.0]);
        let default_emulator = construct_network(EMULATOR_CONFIG, default_emulator_wb);
        let default_generator_wb = generator_factory.forward(&[0.0, 1.0]);
        let default_generator = construct_network(GENERATOR_CONFIG, default_generator_wb);

        // Initial generator samples
        let mut initial_input: Vec<Vec<f32>> =
            (0..NUM_SAMPLES).map(|_| vec![rand::random()]).collect();
        initial_input.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let initial_samples: Vec<f32> = default_generator
            .forward_batch(&initial_input)
            .into_iter()
            .flatten()
            .collect();
        let initial_input: Vec<f64> = initial_input
            .into_iter()
            .flatten()
            .map(|x| x as f64)
            .collect();

        LearningNetworks {
            parabola: ParabolaEmulator {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                emulator_factory,
                current_emulator: default_emulator,
            },
            rng: RandomNumberGenerator {
                mu: 0.0,
                sigma: 1.0,
                generator_factory,
                current_generator: default_generator,
                current_samples: (initial_input, initial_samples),
            },
        }
    }
    fn get_lines(&mut self, changed: bool) -> [Line; 2] {
        const RADIUS: i64 = 512;

        // Construct emulator if changed
        if changed {
            let parameters: [f32; 3] = [self.parabola.a, self.parabola.b, self.parabola.c];
            let emulator_wb: Vec<f32> = self.parabola.emulator_factory.forward(&parameters);
            self.parabola.current_emulator = construct_network(EMULATOR_CONFIG, emulator_wb);
        }
        let emulator: &FullyConnected = &self.parabola.current_emulator;

        // Get grid
        let x: Vec<Vec<f32>> = (-RADIUS..=RADIUS)
            .map(|i| vec![i as f32 / RADIUS as f32])
            .collect();
        let x_double: Vec<f64> = (-RADIUS..=RADIUS)
            .map(|i| i as f64 / RADIUS as f64)
            .collect();

        // Calculate expectation
        let y_expected: Vec<f64> = x
            .iter()
            .flatten()
            .map(|x| ((self.parabola.a * x * x) + (self.parabola.b * x) + self.parabola.c) as f64)
            .collect();

        // Propagate through emulator
        let y: Vec<f32> = emulator.forward_batch(&x).into_iter().flatten().collect();

        // Form pairs
        let expected_pairs: PlotPoints = x_double
            .clone()
            .into_iter()
            .zip(y_expected)
            .map(|(x, y)| [x, y])
            .collect();
        let emualator_pairs: PlotPoints = x_double
            .into_iter()
            .zip(y)
            .map(|(x, y)| [x, y as f64])
            .collect();

        // Create lines
        let expected_line: Line = Line::new(expected_pairs)
            .color(Color32::from_rgb(200, 100, 200))
            // .color(Color32::from_rgb(0, 255, 0))
            .name("expected")
            .width(3.0);
        let emualator_line: Line = Line::new(emualator_pairs)
            .color(Color32::from_rgb(100, 200, 100))
            // .color(Color32::from_rgb(255, 0, 0))
            .name("emulator")
            .width(3.0)
            .style(LineStyle::Dashed { length: 10.0 });

        // return lines
        [expected_line, emualator_line]
    }
    fn get_figures(&mut self, changed: bool, resample: bool) -> (Line, BarChart) {
        // Construct generator and new samples if changed
        if changed {
            let parameters: [f32; 2] = [self.rng.mu, self.rng.sigma];
            let generator_wb = self.rng.generator_factory.forward(&parameters);
            self.rng.current_generator = construct_network(GENERATOR_CONFIG, generator_wb);
        };
        if changed || resample {
            // Get grid (sort shouldn't change results, just plot)
            let mut x: Vec<Vec<f32>> = (0..NUM_SAMPLES).map(|_| vec![rand::random()]).collect();
            x.sort_by(|a, b| a.partial_cmp(&b).unwrap());
            let x_double: Vec<f64> = x.iter().map(|f32_sample| f32_sample[0] as f64).collect();

            // Propagate through generator
            let samples: Vec<f32> = self
                .rng
                .current_generator
                .forward_batch(&x)
                .into_iter()
                .flatten()
                .collect();

            // Update samples
            self.rng.current_samples = (x_double, samples)
        }

        // Create barcharts for samples
        let (ref inputs, ref samples) = self.rng.current_samples;
        let min = self.rng.mu - 4.0 * self.rng.sigma;
        let max = self.rng.mu + 4.0 * self.rng.sigma;
        let generator_bars = histogram(samples, NUM_BARS, min, max);
        let generator_hist: BarChart = BarChart::new(generator_bars)
            .color(Color32::from_rgb(200, 100, 200))
            .name("generator");

        // Create line for expectation
        let expected_pairs: Vec<[f64; 2]> = inputs
            .into_iter()
            .map(|&x| {
                [
                    x * 8.0 * self.rng.sigma as f64 + self.rng.mu as f64
                        - 4.0 * self.rng.sigma as f64,
                    normal(
                        x * 8.0 * self.rng.sigma as f64 + self.rng.mu as f64
                            - 4.0 * self.rng.sigma as f64,
                        self.rng.mu as f64,
                        self.rng.sigma as f64,
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
