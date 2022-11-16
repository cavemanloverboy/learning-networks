use egui::{
    panel::Side,
    plot::{Legend, Line, Plot, PlotPoints},
    Color32,
};
use fabricator::{Factory, Network};
use plotly::{
    common::{Fill, Font, Marker, MarkerSymbol, Mode, Title},
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
    vs.load("emulator.ot").expect("failed to load fabricator");
    println!("fc5 has shape {:?}", factory.fc5.ws.size());

    plot_parabolas(factory);
}

fn color(i: usize) -> NamedColor {
    match i {
        0 => NamedColor::Blue,
        1 => NamedColor::Crimson,
        2 => NamedColor::LimeGreen,
        _ => unreachable!("only 3 parabolas"),
    }
}

fn plot_parabolas(factory: Factory) {
    // Define parabola parameter sets
    const PARAMS: [[f32; 3]; 3] = [[-0.8, -0.3, 0.0], [0.2, 0.2, 0.4], [0.6, -1.0, -0.6]];

    // Formatting
    const FONT_SIZE: usize = 20;

    // Construct grid
    const NUM_PTS: usize = 256;
    let x_grid: Vec<f32> = (0..NUM_PTS)
        .map(|i| i as f32 * 2.0 / NUM_PTS as f32 - 1.0)
        .collect();
    let x_tensor: Tensor = Tensor::of_slice(&x_grid).reshape(&[NUM_PTS as i64, 1]);

    // Create plot
    let mut plot = plotly::Plot::new();

    // Change layout
    let layout = plotly::Layout::new()
        .legend(plotly::layout::Legend::new().font(Font::new().size(FONT_SIZE)))
        .y_axis(
            Axis::new()
                .title(
                    Title::from("Emulator\t Output")
                        .font(Font::new().size(FONT_SIZE))
                        .x(0.0),
                )
                .tick_values(vec![-2.0, -1.0, 0.0, 1.0, 2.0])
                .grid_color(NamedColor::DarkGray)
                .tick_font(Font::new().size(FONT_SIZE)),
        )
        .x_axis(
            Axis::new()
                .title(
                    Title::from("Independent\t Variable")
                        .font(Font::new().size(FONT_SIZE))
                        .x(0.0),
                )
                .tick_values(vec![-1.0, -0.5, 0.0, 0.5, 1.0])
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

    for (i, [a, b, c]) in PARAMS.into_iter().enumerate() {
        // Generate emulator for this parameter set
        let emulator: Network = factory.manufacture_network(&[a, b, c]);

        // Get actual parabola
        let expectation: Vec<f32> = x_grid.iter().map(|&x| (a * x * x) + (b * x) + c).collect();
        let expectation_trace = Scatter::new(x_grid.clone(), expectation)
            .line(plotly::common::Line::new().color(color(i)).width(3.0))
            .name(&format!("P{}", i + 1));

        // Get parabola prediction
        let prediction: Vec<f32> = emulator.forward(&x_tensor).view(-1).into();
        let prediction_trace = Scatter::new(
            x_grid.clone().into_iter().step_by(10),
            prediction.into_iter().step_by(10),
        )
        .mode(Mode::Markers)
        .marker(
            Marker::new()
                .symbol(MarkerSymbol::Cross)
                .size(10)
                .color(color(i)),
        )
        .show_legend(false);

        // Pick color, add traces
        plot.add_trace(expectation_trace);
        plot.add_trace(prediction_trace);
    }

    plot.save("figure1", ImageFormat::PDF, 1024, 680, 1.0);
}
