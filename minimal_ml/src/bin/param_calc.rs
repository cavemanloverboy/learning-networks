use minimal_ml::{network_parameter_calculator, Config};

fn main() {
    let big_network = Config {
        width: 128,
        hidden_layers: 4,
        in_dims: 3,
        out_dims: 1,
    };

    let small_network = Config {
        width: 32,
        hidden_layers: 3,
        in_dims: 3,
        out_dims: 1,
    };

    println!(
        "big {}, small {}",
        network_parameter_calculator(&big_network),
        network_parameter_calculator(&small_network),
    )
}
