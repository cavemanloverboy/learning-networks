use std::ops::Sub;

use super::{Config, FullyConnected};

pub fn construct_network(network_config: Config, mut network_wb: Vec<f32>) -> FullyConnected {
    let total_num_weights = network_num_weights(&network_config);
    let mut network_weights: Vec<f32> = network_wb.drain(0..total_num_weights).collect();
    let mut network_biases = network_wb; // the remainder

    // Allocate for weights and biases
    let mut weights: Vec<Vec<f32>> = Vec::with_capacity(network_config.hidden_layers + 1);
    let mut biases: Vec<Vec<f32>> = Vec::with_capacity(network_config.hidden_layers + 1);

    // Get weights
    let num_first_weights = network_config.width * network_config.in_dims;
    let num_final_weights = network_config.width * network_config.out_dims;
    let first_w: Vec<f32> = network_weights
        .drain(network_weights.len() - num_first_weights..network_weights.len())
        .collect();
    let final_w: Vec<f32> = network_weights
        .drain(network_weights.len() - num_final_weights..network_weights.len())
        .collect();
    weights.push(first_w);
    for _ in 1..network_config.hidden_layers {
        weights.push(
            network_weights
                .drain(network_weights.len() - network_config.width.pow(2)..network_weights.len())
                .collect(),
        );
    }
    weights.push(final_w);

    // Get biases
    let num_first_bias = network_config.width;
    let num_final_bias = network_config.out_dims;
    let first_b: Vec<f32> = network_biases
        .drain(network_biases.len() - num_first_bias..network_biases.len())
        .collect();
    let final_b: Vec<f32> = network_biases
        .drain(network_biases.len() - num_final_bias..network_biases.len())
        .collect();
    biases.push(first_b);
    for _ in 1..network_config.hidden_layers {
        biases.push(
            network_biases
                .drain(network_biases.len() - network_config.width..network_biases.len())
                .collect(),
        );
    }
    biases.push(final_b);

    // Get biases
    assert_eq!(network_biases.len(), 0);
    assert_eq!(network_weights.len(), 0);

    // Zip and construct network
    let network_weights_and_biases = weights.into_iter().zip(biases).collect();
    let network = FullyConnected::new(network_config, network_weights_and_biases);
    network
}

/// Counts the number of network parameters
pub fn network_parameter_calculator(config: &Config) -> usize {
    // First layer
    config.width * config.in_dims  + config.width
    // Inner layers
    + config.hidden_layers.sub(1) * (config.width.pow(2) + config.width)
    // Final layer
    + config.out_dims * config.width + config.out_dims
}

pub fn network_num_weights(config: &Config) -> usize {
    network_parameter_calculator(config) - network_num_biases(config)
}

pub fn network_num_biases(config: &Config) -> usize {
    config.width * config.hidden_layers + config.out_dims
}
