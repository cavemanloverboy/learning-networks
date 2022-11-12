use std::{io::Read, ops::Sub};

use lazy_static::lazy_static;
use linear::{Linear, Vector};
use serde::{Deserialize, Serialize};

use crate::linear::Matrix;

pub mod linear;
pub mod utils;

#[derive(Serialize, Deserialize)]
pub struct FullyConnected {
    config: Config,
    layers: Vec<Linear>,
}

pub fn load_emulator(network_config: &Config) -> FullyConnected {
    let mut bytes = vec![];
    let mut file = std::fs::File::open("emulator.al").unwrap();
    file.read_to_end(&mut bytes).unwrap();
    let weights_and_biases: Vec<(Vec<f32>, Vec<f32>)> = serde_cbor::from_slice(&bytes).unwrap();

    let config = Config {
        width: 128,
        hidden_layers: 4,
        in_dims: 3,
        out_dims: network_parameter_calculator(&network_config),
    };
    FullyConnected::new(config, weights_and_biases)
}

pub fn load_generator(network_config: &Config) -> FullyConnected {
    let mut bytes = vec![];
    let mut file = std::fs::File::open("generator.al").unwrap();
    file.read_to_end(&mut bytes).unwrap();
    let weights_and_biases: Vec<(Vec<f32>, Vec<f32>)> = serde_cbor::from_slice(&bytes).unwrap();

    let config = Config {
        width: 128,
        hidden_layers: 4,
        in_dims: 2,
        out_dims: network_parameter_calculator(&network_config),
    };
    FullyConnected::new(config, weights_and_biases)
}

impl FullyConnected {
    pub fn new(
        config: Config,
        mut weights_and_biases: Vec<(Vec<f32>, Vec<f32>)>,
    ) -> FullyConnected {
        // Check that the number of layers passed in is correct
        assert_eq!(
            weights_and_biases.len(),
            config.hidden_layers + 1,
            "number of weights and biases passed in does not match config"
        );
        assert!(
            config.hidden_layers > 0,
            "must have at least one hidden layer"
        );

        // Allocate for all layers
        let mut layers = Vec::with_capacity(config.hidden_layers + 1);

        // Construct all hidden layers, starting with the first
        let (first_weights, first_biases) = weights_and_biases.remove(0);
        // Check that the number of weights and biases are correct
        assert_eq!(first_weights.len(), config.in_dims * config.width);
        assert_eq!(first_biases.len(), config.width);
        // Then construct weights and biases
        let ws = Matrix {
            data: first_weights,
            in_dim: config.in_dims,
            out_dim: config.width,
        };
        let bs = first_biases;
        // Finally, construct layer
        layers.push(Linear::new(ws, bs));

        // Proceed with all hidden layers
        for l in 1..config.hidden_layers {
            // Get weights and biases
            let (hidden_weights, hidden_biases) = weights_and_biases.remove(0);
            // Check that the number of weights and biases are correct
            assert_eq!(hidden_weights.len(), config.width * config.width);
            assert_eq!(hidden_biases.len(), config.width);
            // Then construct weights and biases
            let ws = Matrix {
                data: hidden_weights,
                in_dim: config.width,
                out_dim: config.width,
            };
            let bs = hidden_biases;
            // Finally, construct layer
            layers.push(Linear::new(ws, bs));
        }

        // Then, construct final layer
        let (final_weights, final_biases) = weights_and_biases.remove(0);
        // Check that the number of weights and biases are correct
        assert_eq!(final_weights.len(), config.width * config.out_dims);
        assert_eq!(final_biases.len(), config.out_dims);
        // Then construct weights and biases
        let ws = Matrix {
            data: final_weights,
            in_dim: config.width,
            out_dim: config.out_dims,
        };
        let bs = final_biases;
        // Finally, construct layer
        layers.push(Linear::new(ws, bs));

        FullyConnected { config, layers }
    }

    pub fn forward_batch(&self, x: &Vec<Vector>) -> Vec<Vector> {
        match self.config.hidden_layers {
            0 => self.layers[0].forward_batch(x),
            _ => {
                // first layer
                let mut output = self.layers[0].forward_batch_relu(x);
                for l in 1..self.layers.len().sub(1) {
                    // middle layers
                    output = self.layers[l].forward_batch_relu(&output);
                }
                // output layer
                self.layers.last().unwrap().forward_batch(&output)
            }
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vector {
        match self.config.hidden_layers {
            0 => self.layers[0].forward(x),
            _ => {
                // first layer
                let mut output = self.layers[0].forward_relu(x);
                for l in 1..self.layers.len().sub(1) {
                    // middle layers
                    output = self.layers[l].forward_relu(&output);
                }
                // output layer
                self.layers.last().unwrap().forward(&output)
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Config {
    pub width: usize,
    pub hidden_layers: usize,
    pub in_dims: usize,
    pub out_dims: usize,
}

#[test]
#[should_panic]
fn test_no_hidden() {
    // Construct config for pre-trained network
    let config = Config {
        width: 32,
        hidden_layers: 0,
        in_dims: 1,
        out_dims: 1,
    };

    // Mock pretrained network
    let mut weights_and_biases = Vec::with_capacity(1);
    let output_weights = vec![1.0; config.in_dims * config.out_dims];
    let output_biases = vec![1.0; config.out_dims];
    weights_and_biases.push((output_weights, output_biases));

    // Construct network
    let network = FullyConnected::new(config, weights_and_biases);

    // Let's try some input
    let input = vec![vec![1.0], vec![2.0]];
    let _output = network.forward_batch(&input);
    let input = vec![1.0];
    let _output = network.forward(&input);
}

#[test]
fn test_one_hidden() {
    // Construct config for pre-trained network
    let config = Config {
        width: 32,
        hidden_layers: 1,
        in_dims: 1,
        out_dims: 1,
    };

    // Mock pretrained network
    let mut weights_and_biases = Vec::with_capacity(1);
    let first_weights = vec![1.0; config.in_dims * config.width];
    let first_biases = vec![1.0; config.width];
    weights_and_biases.push((first_weights, first_biases));

    let output_weights = vec![1.0; config.width * config.out_dims];
    let output_biases = vec![1.0; config.out_dims];
    weights_and_biases.push((output_weights, output_biases));
    // Construct network
    let network = FullyConnected::new(config, weights_and_biases);

    // Let's try some input
    let input = vec![vec![1.0], vec![2.0]];
    let _output = network.forward_batch(&input);
    let input = vec![1.0];
    let _output = network.forward(&input);
}

#[test]
fn test_two_hidden() {
    // Construct config for pre-trained network
    let config = Config {
        width: 32,
        hidden_layers: 2,
        in_dims: 1,
        out_dims: 1,
    };

    // Mock pretrained network
    let mut weights_and_biases = Vec::with_capacity(1);
    let first_weights = vec![1.0; config.in_dims * config.width];
    let first_biases = vec![1.0; config.width];
    weights_and_biases.push((first_weights, first_biases));

    let second_weights = vec![1.0; config.width * config.width];
    let second_biases = vec![1.0; config.width];
    weights_and_biases.push((second_weights, second_biases));

    let output_weights = vec![1.0; config.width * config.out_dims];
    let output_biases = vec![1.0; config.out_dims];
    weights_and_biases.push((output_weights, output_biases));

    // Construct network
    let network = FullyConnected::new(config, weights_and_biases);

    // Let's try some input
    let input = vec![vec![1.0], vec![2.0]];
    let _output = network.forward_batch(&input);
    let input = vec![1.0];
    let _output = network.forward(&input);
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
