use std::ops::Sub;

use minimal_ml::{linear::Vector, load_emulator, utils::construct_network, Config, FullyConnected};
use plotly::{
    common::{Line, Marker, MarkerSymbol, Mode},
    ImageFormat, Plot, Scatter,
};

#[test]
fn load_emulator_factory() {
    let emulator_config = Config {
        width: 64,
        hidden_layers: 3,
        in_dims: 1,
        out_dims: 1,
    };
    let factory: FullyConnected = load_emulator(&emulator_config);

    let input = vec![0.6, -0.2, -0.1];
    let emulator_wb: Vec<f32> = factory.forward_batch(&vec![input.clone()]).remove(0);

    let emulator: FullyConnected = construct_network(emulator_config, emulator_wb);
    // let total_num_weights = emulator_num_weights(&emulator_config);
    // let mut emulator_weights: Vec<f32> = emulator_wb.drain(0..total_num_weights).collect();
    // let mut emulator_biases = emulator_wb; // the remainder

    // // Allocate for weights and biases
    // let mut weights: Vec<Vec<f32>> = Vec::with_capacity(emulator_config.hidden_layers + 1);
    // let mut biases: Vec<Vec<f32>> = Vec::with_capacity(emulator_config.hidden_layers + 1);

    // // Get weights
    // let num_first_weights = emulator_config.width * emulator_config.in_dims;
    // let num_final_weights = emulator_config.width * emulator_config.out_dims;
    // let first_w: Vec<f32> = emulator_weights
    //     .drain(emulator_weights.len() - num_first_weights..emulator_weights.len())
    //     .collect();
    // let final_w: Vec<f32> = emulator_weights
    //     .drain(emulator_weights.len() - num_final_weights..emulator_weights.len())
    //     .collect();
    // weights.push(first_w);
    // for _ in 1..emulator_config.hidden_layers {
    //     weights.push(
    //         emulator_weights
    //             .drain(
    //                 emulator_weights.len() - emulator_config.width.pow(2)..emulator_weights.len(),
    //             )
    //             .collect(),
    //     );
    // }
    // weights.push(final_w);

    // // Get biases
    // let num_first_bias = emulator_config.width;
    // let num_final_bias = emulator_config.out_dims;
    // let first_b: Vec<f32> = emulator_biases
    //     .drain(emulator_biases.len() - num_first_bias..emulator_biases.len())
    //     .collect();
    // let final_b: Vec<f32> = emulator_biases
    //     .drain(emulator_biases.len() - num_final_bias..emulator_biases.len())
    //     .collect();
    // biases.push(first_b);
    // for _ in 1..emulator_config.hidden_layers {
    //     biases.push(
    //         emulator_biases
    //             .drain(emulator_biases.len() - emulator_config.width..emulator_biases.len())
    //             .collect(),
    //     );
    // }
    // biases.push(final_b);

    // // Get biases
    // assert_eq!(emulator_biases.len(), 0);
    // assert_eq!(emulator_weights.len(), 0);

    // // Zip and construct emulator
    // let emulator_weights_and_biases = weights.into_iter().zip(biases).collect();
    // let emulator = FullyConnected::new(emulator_config, emulator_weights_and_biases);

    let xs: Vec<Vector> = (0..1001)
        .map(|i| vec![(i as f32 / 1001.0 * 2.0) - 1.0])
        .collect();
    let output: Vec<f32> = emulator.forward_batch(&xs).into_iter().flatten().collect();
    let exp: Vec<f32> = xs
        .iter()
        .flatten()
        .map(|x| input[0] * x * x + input[1] * x + input[2])
        .collect();
    let xs: Vec<f32> = xs.into_iter().flatten().collect();
    let trace = Scatter::new(
        xs.iter().cloned().step_by(50),
        output.into_iter().step_by(50),
    )
    .mode(Mode::Markers)
    .marker(Marker::new().symbol(MarkerSymbol::Cross).size(10));
    let trace2 = Scatter::new(xs, exp).line(Line::new().width(3.0));

    let mut plot = Plot::new();
    plot.add_trace(trace2);
    plot.add_trace(trace);
    plot.save("test", ImageFormat::PNG, 1024, 680, 1.0);
}
