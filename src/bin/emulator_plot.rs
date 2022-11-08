use egui::{
    panel::Side,
    plot::{Legend, Line, Plot, PlotPoints},
    Color32,
};
use fabricator::{Factory, Network};
use plotly::{
    common::{Font, Title},
    layout::{Axis, AxisType},
    ImageFormat, NamedColor, Scatter,
};
use tch::{nn, Tensor};
fn main() {
    // Load variables, factory from disk
    let device = tch::Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device);

    // Define nn parameters. TODO: store in vs or elsewhere
    let emulator_input: i64 = 1;
    let emulator_width: i64 = 16;
    let emulator_depth: i64 = 1;
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
    vs.load("emulator.ot").expect("failed to load fabricator");

    plot_parabolas(factory);
}

fn plot_parabolas(factory: Factory) {
    // Define parabola parameter sets
    const PARAMS: [[f32; 3]; 6] = [
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, 0.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, -1.0],
        [1.0, 0.0, 0.0],
        [1.0, -1.0, 1.0],
    ];

    // Formatting
    // const COLORS
    const FONT_SIZE: usize = 16;

    // Construct grid
    const NUM_PTS: usize = 256;
    let x_grid: Vec<f32> = (0..NUM_PTS).map(|i| i as f32 * 2.0 - 1.0).collect();
    let x_tensor: Tensor = Tensor::of_slice(&x_grid);

    // Create plot
    let mut plot = plotly::Plot::new();

    for (i, [a, b, c]) in PARAMS.into_iter().enumerate() {
        // Generate emulator for this parameter set
        let emulator: Network = factory.manufacture_network(&[a, b, c]);

        // Get actual parabola
        let expectation: Vec<f32> = x_grid.iter().map(|&x| (a * x * x) + (b * x) + c).collect();
        let expectation_trace = Scatter::new(x_grid.clone(), expectation);

        // Get parabola prediction
        let prediction: Vec<f32> = emulator.forward(&x_tensor).view(-1).into();
        let prediction_trace = Scatter::new(x_grid.clone(), prediction);

        // Pick color, add traces
        plot.add_trace(expectation_trace);
        plot.add_trace(prediction_trace);
    }

    // Change layout
    let layout = plotly::Layout::new()
        .y_axis(
            Axis::new()
                .title(Title::from("y(x)").font(Font::new().size(FONT_SIZE)).x(0.0))
                .type_(AxisType::Log)
                .tick_values(vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
                .grid_color(NamedColor::DarkGray),
        )
        .x_axis(
            Axis::new()
                .title(Title::from("x").font(Font::new().size(FONT_SIZE)).x(0.0))
                .tick_values(vec![-1.0, -0.5, 0.0, 0.5, 1.0])
                .type_(AxisType::Log)
                .grid_color(NamedColor::DarkGray),
        );
    plot.set_layout(layout);

    plot.save("asdf", ImageFormat::PNG, 1024, 680, 1.0);
}
