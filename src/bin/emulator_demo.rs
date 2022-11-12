use egui::{
    panel::Side,
    plot::{Legend, Line, LineStyle, Plot, PlotPoints},
    Color32,
};
use fabricator::{Factory, Network};
use tch::{nn, Tensor};
fn main() {
    println!("starting");
    // Load variables, factory from disk
    let device = tch::Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);

    // Define nn parameters. TODO: store in vs or elsewhere
    let emulator_input: i64 = 1;
    let emulator_width: i64 = 64;
    let emulator_depth: i64 = 2;
    let emulator_output: i64 = 1;
    let n_params: i64 = 3;
    let factory = Factory::new(
        &vs.root(),
        n_params,
        emulator_input,
        emulator_depth,
        emulator_width,
        emulator_output,
    );
    println!("loading emulator");
    vs.load("emulator.ot").expect("failed to load fabricator");
    println!("loaded emulator");

    start_up_demo(factory);
}

fn start_up_demo(factory: Factory) {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui App",
        options,
        Box::new(|_cc| Box::new(Parabola::new(factory))),
    );
}

struct Parabola {
    a: f32,
    b: f32,
    c: f32,
    factory: Factory,
}

impl eframe::App for Parabola {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Learning Networks Demo");
            ui.label("Choose parabola parameters");
            ui.add(egui::Slider::new(&mut self.a, -1.0..=1.0).text("a"));
            ui.add(egui::Slider::new(&mut self.b, -1.0..=1.0).text("b"));
            ui.add(egui::Slider::new(&mut self.c, -1.0..=1.0).text("c"));
            ui.label(format!(
                "a = {:.2}; b = {:.2}; c = {:.2}",
                self.a, self.b, self.c
            ));
        });
        egui::TopBottomPanel::bottom("bottom").show(ctx, |ui| {
            ui.heading("sick bottom panel bro");
        });
        egui::SidePanel::new(Side::Right, "plot area")
            .min_width(500.0)
            .show(ctx, |ui| {
                let plot = Plot::new("parabola")
                    .legend(Legend::default())
                    .view_aspect(1.0)
                    .include_y(-2.1 + self.c)
                    .include_y(2.1 + self.c);
                plot.show(ui, |plot_ui| {
                    let [expected_line, emulator_line] = self.get_lines();
                    plot_ui.line(expected_line);
                    plot_ui.line(emulator_line);
                });
            });
    }
}

impl Parabola {
    fn new(factory: Factory) -> Self {
        Parabola {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            factory,
        }
    }
    fn get_lines(&self) -> [Line; 2] {
        const RADIUS: i64 = 512;

        // Construct emulator
        let parameters: [f32; 3] = [self.a, self.b, self.c];
        let emulator: Network = self.factory.manufacture_network(&parameters);

        // Get grid
        let x: Vec<f32> = (-RADIUS..=RADIUS)
            .map(|i| i as f32 / RADIUS as f32)
            .collect();
        let x_double: Vec<f64> = (-RADIUS..=RADIUS)
            .map(|i| i as f64 / RADIUS as f64)
            .collect();
        let x_tensor: Tensor = Tensor::of_slice(&x).reshape(&[2 * RADIUS + 1, 1]);

        // Calculate expectation
        let y_expected: Vec<f64> = x
            .iter()
            .map(|x| ((self.a * x * x) + (self.b * x) + self.c) as f64)
            .collect();

        // Propagate through emulator
        let y_tensor: Tensor = emulator.forward(&x_tensor);
        let y = Vec::<f32>::from(y_tensor.contiguous().view(-1));

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
            .width(2.0);
        let emualator_line: Line = Line::new(emualator_pairs)
            .color(Color32::from_rgb(100, 200, 100))
            // .color(Color32::from_rgb(255, 0, 0))
            .name("emulator")
            .width(2.0)
            .style(LineStyle::Dashed { length: 10.0 });

        // return lines
        [expected_line, emualator_line]
    }
}
