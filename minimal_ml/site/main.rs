use std::f64::consts::PI;

use egui::{
    panel::Side,
    plot::{Bar, BarChart, Legend, Line, LineStyle, Plot, PlotPoints},
    Color32, FontFamily, FontId, RichText, ScrollArea,
};
use minimal_ml::{
    construct_emulator_factory, construct_generator_factory, utils::construct_network, Config,
    FullyConnected,
};

const EMULATOR_CONFIG: Config = Config {
    width: 64,
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

const POST: &'static str = include_str!("post.txt");

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| Box::new(LearningNetworks::new())),
    );
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Make sure panics are logged using `console.error`.
    console_error_panic_hook::set_once();

    // Redirect tracing to console.log and friends:
    tracing_wasm::set_as_global_default();

    let options = eframe::WebOptions::default();

    println!("starting web");
    eframe::start_web(
        "learningnetworks",
        options,
        Box::new(|_cc| Box::new(LearningNetworks::new())),
    )
    .expect("failed to start eframe");
}

struct LearningNetworks {
    parabola: ParabolaEmulator,
    rng: RandomNumberGenerator,
}

struct ParabolaEmulator {
    a: f32,
    b: f32,
    c: f32,
    emulator_factory: FullyConnected,
    // plot_bytes: Vec<u8>,
}
struct RandomNumberGenerator {
    mu: f32,
    sigma: f32,
    generator_factory: FullyConnected,
}

impl eframe::App for LearningNetworks {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // egui::TopBottomPanel::bottom("easy_mark_bottom").show(ctx, |ui| {
            //     let layout = egui::Layout::top_down(egui::Align::Center).with_main_justify(true);
            //     ui.allocate_ui_with_layout(ui.available_size(), layout, |ui| {
            //         ui.heading("Learning Networks: Ultralight Emulators, Generators, and Beyond (Zamora et. al. 2022)");
            //     })
            // });
            ui.columns(2, |columns| {
                // Column 1: Text
                let changed: bool = ScrollArea::vertical()
                    .id_source("Text")
                    .show(&mut columns[0], |ui| {
                        // Title of page
                        ui.heading(
                            RichText::new("Learning Networks")
                                .font(FontId::proportional(32.0))
                                .color(Color32::WHITE),
                        );

                        // Body
                        ui.label(
                            RichText::new(POST)
                                // .font(FontId::new(16.0, FontFamily::Name("serif".into())))
                                .color(Color32::WHITE),
                        );

                        // Parabola Sliders
                        ui.label("Choose parabola parameters");
                        let a =
                            ui.add(egui::Slider::new(&mut self.parabola.a, -1.0..=1.0).text("a"));
                        let b =
                            ui.add(egui::Slider::new(&mut self.parabola.b, -1.0..=1.0).text("b"));
                        let c =
                            ui.add(egui::Slider::new(&mut self.parabola.c, -1.0..=1.0).text("c"));
                        ui.label(format!(
                            "a = {:.2}; b = {:.2}; c = {:.2}",
                            self.parabola.a, self.parabola.b, self.parabola.c
                        ));

                        // RNG sliders
                        ui.label("\nChoose rng parameters");
                        let mu =
                            ui.add(egui::Slider::new(&mut self.rng.mu, -5.0..=5.0).text("mean"));
                        let sigma =
                            ui.add(egui::Slider::new(&mut self.rng.sigma, 0.5..=2.0).text("std"));
                        ui.label(format!(
                            "mean = {:.2}; std = {:.2}",
                            self.rng.mu, self.rng.sigma,
                        ));

                        a.changed() || b.changed() || c.changed() || mu.changed() || sigma.changed()
                    })
                    .inner;

                // Column 2: Demo area
                // egui::containers::Frame::default().show(&mut columns[1], |ui| {
                // //     .frame(columns[1])
                // columns[1]
                //     // egui::SidePanel::new(Side::Right, "Demos")
                //     // .min_width(500.0)
                //     .show(ctx, |ui| {
                // Get panel dimensions
                // let height = ui.available_height();
                // let width = ui.available_width();
                // let half_height = height / 2.0;

                // // On top half put parabola
                // let plot = Plot::new("parabola")
                //     .legend(Legend::default())
                //     .include_y(-2.1 + self.parabola.c)
                //     .include_y(2.1 + self.parabola.c)
                //     .width(width)
                //     .height(half_height);
                // plot.show(ui, |plot_ui| {
                //     let [expected_line, emulator_line] = self.get_lines();
                //     plot_ui.line(expected_line);
                //     plot_ui.line(emulator_line);
                // });

                // let plot = Plot::new("Generator Demo")
                //     .legend(Legend::default())
                //     .include_x(-9.0)
                //     .include_x(9.0)
                //     .include_y(1.0)
                //     .width(width)
                //     .height(half_height);
                // plot.show(ui, |plot_ui| {
                //     let (expected_line, generator_hist) = self.get_figures();
                //     plot_ui.bar_chart(generator_hist);
                //     plot_ui.line(expected_line);
                // });
                // });
            });
        });

        // egui::SidePanel::new(Side::Right, "Demos")
        //     .min_width(500.0)
        //     .show(ctx, |ui| {
        //         // Get panel dimensions
        //         let height = ui.available_height();
        //         let width = ui.available_width();
        //         let half_height = height / 2.0;

        //         // On top half put parabola
        //         let plot = Plot::new("parabola")
        //             .legend(Legend::default())
        //             .view_aspect(1.0)
        //             .include_y(-2.1 + self.parabola.c)
        //             .include_y(2.1 + self.parabola.c);
        //         plot.show(ui, |plot_ui| {
        //             let [expected_line, emulator_line] = self.get_lines();
        //             plot_ui.line(expected_line);
        //             plot_ui.line(emulator_line);
        //         });
        //     });
    }
}

impl LearningNetworks {
    fn new() -> Self {
        // Load emulator and generator
        let emulator_factory = construct_emulator_factory(&EMULATOR_CONFIG);
        let generator_factory = construct_generator_factory(&GENERATOR_CONFIG);
        LearningNetworks {
            parabola: ParabolaEmulator {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                emulator_factory,
            },
            rng: RandomNumberGenerator {
                mu: 0.0,
                sigma: 1.0,
                generator_factory,
            },
        }
    }
    fn get_lines(&self) -> [Line; 2] {
        const RADIUS: i64 = 512;

        // Construct emulator
        let parameters: [f32; 3] = [self.parabola.a, self.parabola.b, self.parabola.c];
        let emulator_wb: Vec<f32> = self.parabola.emulator_factory.forward(&parameters);
        let emulator: FullyConnected = construct_network(EMULATOR_CONFIG, emulator_wb);

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
    fn get_figures(&self) -> (Line, BarChart) {
        // must be even, otherwise change bar positions
        const NUM_BARS: i64 = 32;
        const NUM_SAMPLES: i64 = 10_000;

        // Construct generator
        let parameters: [f32; 2] = [self.rng.mu, self.rng.sigma];
        let generator_wb = self.rng.generator_factory.forward(&parameters);
        let generator: FullyConnected = construct_network(GENERATOR_CONFIG, generator_wb);

        // Get grid (sort shouldn't change results, just plot)
        let mut x: Vec<Vec<f32>> = (0..NUM_SAMPLES)
            .map(|i| vec![i as f32 / NUM_SAMPLES as f32])
            .collect();
        x.sort_by(|a, b| a.partial_cmp(&b).unwrap());
        let x_double: Vec<f64> = x.iter().map(|f32_sample| f32_sample[0] as f64).collect();

        // Propagate through generator
        let samples: Vec<f32> = generator.forward_batch(&x).into_iter().flatten().collect();

        // Create barcharts for samples
        let min = self.rng.mu - 4.0 * self.rng.sigma;
        let max = self.rng.mu + 4.0 * self.rng.sigma;
        let generator_bars = histogram(&samples, NUM_BARS, min, max);
        let generator_hist: BarChart = BarChart::new(generator_bars)
            .color(Color32::from_rgb(200, 100, 200))
            .name("generator");

        // Create line for expectation
        let expected_pairs: Vec<[f64; 2]> = x_double
            .into_iter()
            .map(|x| {
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
