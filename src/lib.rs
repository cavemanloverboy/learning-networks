use std::convert::TryInto;

use tch::{nn, nn::Module, Kind, Tensor};

pub struct Factory {
    pub fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
    fc4: nn::Linear,
    fc5: nn::Linear,
    n_params: i64,
    emulator_input: i64,
    emulator_depth: i64,
    emulator_width: i64,
    emulator_output: i64,
}

pub struct Network {
    /// Input layer (input x width)
    first_layer: nn::Linear,

    /// Inner layers (width x width)
    inner_layers: Vec<nn::Linear>,

    /// Output Layer (width x output)
    last_layer: nn::Linear,
}

impl Network {
    pub fn forward(&self, emulator_input: &Tensor) -> Tensor {
        // Propagate through first layer
        let mut output = self.first_layer.forward(emulator_input).relu();

        // Propagate through inner layers
        for layer in &self.inner_layers {
            output = layer.forward(&output).relu();
        }

        // Propagate through final layer
        self.last_layer.forward(&output)
    }
}

impl Factory {
    // TODO: ensure all things are positive, nonzero
    // TODO: ensure all i64s are the same order in all functions
    pub fn new(
        vs: &nn::Path,
        n_params: i64,
        emulator_input: i64,
        emulator_depth: i64,
        emulator_width: i64,
        emulator_output: i64,
    ) -> Self {
        Factory {
            fc1: nn::linear(vs / "fc1", n_params, 128, Default::default()),
            fc2: nn::linear(vs / "fc2", 128, 128, Default::default()),
            fc3: nn::linear(vs / "fc3", 128, 128, Default::default()),
            fc4: nn::linear(vs / "fc4", 128, 128, Default::default()),
            fc5: nn::linear(
                vs / "fc5",
                128,
                emulator_parameter_calculator(
                    emulator_input,
                    emulator_width,
                    emulator_depth,
                    emulator_output,
                ),
                Default::default(),
            ),
            n_params,
            emulator_output,
            emulator_depth,
            emulator_width,
            emulator_input,
        }
    }

    // Takes in parameters, and produces an emulator for that subspace.
    pub fn manufacture_network(&self, parameters: &[f32]) -> Network {
        // Run parameter input through factory to get weights and biases
        let [weights, biases]: [Tensor; 2] = self.calculate_weights_and_biases(parameters);

        // Construct emulator through weights and biases
        let emulator: Network = self.construct_emulator(weights, biases);

        emulator
    }

    pub fn calculate_weights_and_biases(&self, parameters: &[f32]) -> [Tensor; 2] {
        // Run parameters through all factory layers
        let layer1_output: Tensor = self
            .fc1
            .forward(&Tensor::of_slice(parameters).view([-1, self.n_params]))
            .relu();
        let layer2_output: Tensor = self.fc2.forward(&layer1_output).relu();
        let layer3_output: Tensor = self.fc3.forward(&layer2_output).relu();
        let layer4_output: Tensor = self.fc4.forward(&layer3_output).relu();
        let layer5_output: Tensor = self.fc5.forward(&layer4_output);

        // Calculate number of weights and biases and split last layer
        let [num_weights, num_biases]: [i64; 2] = self.emulator_num_weights_and_biases();
        let weights_and_biases = layer5_output.split_sizes(&[num_weights, num_biases], 1);

        weights_and_biases
            .try_into()
            .expect("Vec<Tensor> should have length 2")
    }

    pub fn emulator_num_weights_and_biases(&self) -> [i64; 2] {
        // Calculate number of weights and biases
        [
            emulator_num_weights(
                self.emulator_input,
                self.emulator_width,
                self.emulator_depth,
                self.emulator_output,
            ),
            emulator_num_biases(
                self.emulator_input,
                self.emulator_width,
                self.emulator_depth,
                self.emulator_output,
            ),
        ]
    }

    pub fn construct_emulator(&self, weights: Tensor, biases: Tensor) -> Network {
        // Get total number of weights and biases
        let total_weights = weights.size2().unwrap().1;
        let total_biases = biases.size2().unwrap().1;
        let mut weights_used = 0;
        let mut biases_used = 0;

        // Initialize splitting tensors
        let mut weight_split = weights.split_sizes(
            &[
                total_weights - self.emulator_width * self.emulator_input,
                self.emulator_width * self.emulator_input,
            ],
            1,
        );
        let mut bias_split = biases.split_sizes(
            &[total_biases - self.emulator_width, self.emulator_width],
            1,
        );
        weights_used += self.emulator_width * self.emulator_input;
        biases_used += self.emulator_width;

        // Get weights and biases for first layer
        let first_ws = weight_split
            .split_off(1)
            .swap_remove(0)
            .reshape(&[self.emulator_width, self.emulator_input]);
        let first_bs = Some(
            bias_split
                .split_off(1)
                .swap_remove(0)
                .reshape(&[self.emulator_width]),
        );

        // Construct first layer
        let first_layer = nn::Linear {
            ws: first_ws,
            bs: first_bs,
        };

        // Perform split for last layer
        weight_split = weight_split[0].split_sizes(
            &[
                total_weights - self.emulator_output * self.emulator_width - weights_used,
                self.emulator_output * self.emulator_width,
            ],
            1,
        );
        bias_split = bias_split[0].split_sizes(
            &[
                total_biases - self.emulator_output - biases_used,
                self.emulator_output,
            ],
            1,
        );
        weights_used += self.emulator_output * self.emulator_width;
        biases_used += self.emulator_output;

        // Get weights and biases for last layer
        let last_ws = weight_split
            .split_off(1)
            .swap_remove(0)
            .reshape(&[self.emulator_output, self.emulator_width]);
        let last_bs = Some(
            bias_split
                .split_off(1)
                .swap_remove(0)
                .reshape(&[self.emulator_output]),
        );

        // Construct last layer
        let last_layer = nn::Linear {
            ws: last_ws,
            bs: last_bs,
        };

        // Construct inner layers
        let mut inner_layers = Vec::with_capacity(self.emulator_depth as usize);
        for _layer in 0..self.emulator_depth {
            // Check if this is remainder of weights and biases
            let not_final = total_weights - self.emulator_width.pow(2) - weights_used > 0;

            let ws;
            let bs;
            if not_final {
                // Update split if not final
                weight_split = weight_split.swap_remove(0).split_sizes(
                    &[
                        total_weights - self.emulator_width.pow(2) - weights_used,
                        self.emulator_width.pow(2),
                    ],
                    1,
                );

                bias_split = bias_split.swap_remove(0).split_sizes(
                    &[
                        total_biases - self.emulator_width - biases_used,
                        self.emulator_width,
                    ],
                    1,
                );

                // Retrieve weights and biases
                ws = weight_split
                    .split_off(1)
                    .swap_remove(0)
                    .reshape(&[self.emulator_width, self.emulator_width]);
                bs = Some(bias_split.split_off(1).swap_remove(0));
            } else {
                // Retrieve weights and biases
                ws = weight_split
                    .swap_remove(0)
                    .reshape(&[self.emulator_width, self.emulator_width]);
                bs = Some(bias_split.swap_remove(0));
            }

            // Update counters
            weights_used += self.emulator_width.pow(2);
            biases_used += self.emulator_width;

            // Construct layer
            inner_layers.push(nn::Linear { ws, bs })
        }

        // Construct emulator
        let emulator = Network {
            first_layer,
            inner_layers,
            last_layer,
        };

        // Sanity Check
        assert_eq!(weights_used, total_weights);
        assert_eq!(biases_used, total_biases);

        emulator
    }
}

// Reconstruction + KL divergence losses summed over all elements and batch dimension.
pub fn loss(output: &Tensor, x: &Tensor, params: [f32; 3]) -> Tensor {
    let expected = params[2] + (params[1] * x) + (params[0] * x * x);
    (output - expected).pow_tensor_scalar(2).sum(Kind::Double)
}

/// Counts the number of emulator parameters
pub fn emulator_parameter_calculator(
    emulator_input: i64,
    emulator_width: i64,
    emulator_depth: i64,
    emulator_output: i64,
) -> i64 {
    // First layer
    emulator_width * emulator_input  + emulator_width
    // Inner layers
    + emulator_depth * (emulator_width.pow(2) + emulator_width)
    // Final layer
    + emulator_output * emulator_width + emulator_output
}

pub fn emulator_num_weights(
    emulator_input: i64,
    emulator_width: i64,
    emulator_depth: i64,
    emulator_output: i64,
) -> i64 {
    emulator_parameter_calculator(
        emulator_input,
        emulator_width,
        emulator_depth,
        emulator_output,
    ) - emulator_num_biases(
        emulator_input,
        emulator_width,
        emulator_depth,
        emulator_output,
    )
}

pub fn emulator_num_biases(
    _emulator_input: i64,
    emulator_width: i64,
    emulator_depth: i64,
    emulator_output: i64,
) -> i64 {
    emulator_width * (emulator_depth + 1) + emulator_output
}

pub fn gen_batch(size: i64) -> (Tensor, [f32; 3]) {
    // Initialize coefficients
    let [a, b, c]: [f32; 3] = rand::random::<[f32; 3]>().map(|r| 2.0 * r - 1.0);

    // Generate random inputs
    let x: Vec<f32> = (0..size)
        .map(|_| rand::random::<f32>() * 2.0 - 1.0)
        .collect();
    let x = Tensor::of_slice(&x).reshape(&[size, 1]);

    (x, [a, b, c])
}

pub fn gen_uniforms(size: i64) -> Tensor {
    // Generate random inputs
    let x: Vec<f32> = (0..size).map(|_| rand::random::<f32>()).collect();
    Tensor::of_slice(&x).reshape(&[size, 1])
}

pub fn gen_batch_double(size: i64) -> (Tensor, [f64; 3]) {
    // Initialize coefficients
    let [a, b, c]: [f64; 3] = rand::random();

    // Generate random inputs
    let x: Vec<f64> = (0..size)
        .map(|_| rand::random::<f64>() * 2.0 - 1.0)
        .collect();
    let x = Tensor::of_slice(&x).reshape(&[size, 1]);

    (x, [a, b, c])
}

#[test]
fn test_no_inner() {
    let emulator_input: i64 = 1;
    let emulator_width: i64 = 3;
    let emulator_depth: i64 = 0;
    let emulator_output: i64 = 1;
    assert_eq!(
        emulator_parameter_calculator(
            emulator_input,
            emulator_width,
            emulator_depth,
            emulator_output
        ),
        10 // (3 * 1 + 3) + (1 * 3 + 1)
    );
    assert_eq!(
        emulator_num_biases(
            emulator_input,
            emulator_width,
            emulator_depth,
            emulator_output
        ),
        4
    );
    assert_eq!(
        emulator_num_weights(
            emulator_input,
            emulator_width,
            emulator_depth,
            emulator_output
        ),
        6
    );
}

#[test]
fn test_one_inner() {
    let emulator_input: i64 = 1;
    let emulator_width: i64 = 3;
    let emulator_depth: i64 = 1;
    let emulator_output: i64 = 1;
    assert_eq!(
        emulator_parameter_calculator(
            emulator_input,
            emulator_width,
            emulator_depth,
            emulator_output
        ),
        22
    );
}

#[test]
fn test_two_inner() {
    let emulator_input: i64 = 1;
    let emulator_width: i64 = 3;
    let emulator_depth: i64 = 2;
    let emulator_output: i64 = 1;
    assert_eq!(
        emulator_parameter_calculator(
            emulator_input,
            emulator_width,
            emulator_depth,
            emulator_output
        ),
        22 + 9 + 3
    );
}

#[test]
fn test_tensor_split() {
    let tensor = Tensor::of_slice(&[1.0_f32, 2.0_f32]).reshape(&[1, 2]);
    let split = tensor.split_sizes(&[1, 1], 1);
    let split_array: [Tensor; 2] = split.try_into().unwrap();
    assert_eq!(split_array.len(), 2);
    assert_eq!(split_array[0].size(), vec![1, 1]);
    assert_eq!(split_array[1].size(), vec![1, 1]);
    assert_eq!(tensor.size(), vec![1, 2]);
}

#[test]
fn test_tensor_split_1x4() {
    let tensor = Tensor::of_slice(&[1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32]).reshape(&[1, 4]);
    let split = tensor.split_sizes(&[3, 1], 1);
    let split_array: [Tensor; 2] = split.try_into().unwrap();
    assert_eq!(split_array.len(), 2);
    assert_eq!(split_array[0].size(), vec![1, 3]);
    assert_eq!(split_array[1].size(), vec![1, 1]);
    assert_eq!(tensor.size(), vec![1, 4]);
}

#[test]
fn test_manufacture_emulator() {
    let emulator_input: i64 = 1;
    let emulator_width: i64 = 3;
    let emulator_depth: i64 = 0;
    let emulator_output: i64 = 1;
    let n_params: i64 = 1;

    // Create factory
    let device = tch::Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let factory = Factory::new(
        &vs.root(),
        n_params,
        emulator_input,
        emulator_depth,
        emulator_width,
        emulator_output,
    );

    // Construct Emulator
    let parameters = [1.0_f32];
    let emulator: Network = factory.manufacture_network(&parameters);

    // Propagate an input through emulator
    let input = Tensor::of_slice(&[1.0_f32]);
    let output = emulator.forward(&input);
    println!("emulator output is {output:?}");
}

// #[test]
// fn test_manufacture_emulator_double() {
//     let emulator_input: i64 = 1;
//     let emulator_width: i64 = 3;
//     let emulator_depth: i64 = 2;
//     let emulator_output: i64 = 1;
//     let n_params: i64 = 1;

//     // Create factory
//     let device = tch::Device::Cpu;
//     let mut vs = nn::VarStore::new(device);
//     let factory = Factory::new(
//         &vs.root(),
//         n_params,
//         emulator_input,
//         emulator_depth,
//         emulator_width,
//         emulator_output,
//     );

//     vs.double();

//     // Construct Emulator
//     let parameters = [1.0_f32];
//     let emulator: Emulator = factory.manufacture_emulator(&parameters);

//     // Propagate an input through emulator
//     let input = Tensor::of_slice(&[1.0_f64]);
//     let output = emulator.forward(&input);
//     println!("emulator output is {output:?}");
// }

#[test]
fn test_slice() {
    let tensor_1 = Tensor::of_slice(&[1.0_f32, 2.0, 3.0]);
    let expected = Tensor::of_slice(&[1.0_f32, 1.0]);
    assert_eq!(
        tensor_1.slice(0, 1, None, 1) - tensor_1.slice(0, None, -1, 1),
        expected
    )
}

#[test]
fn test_diff() {
    let tensor_1 = Tensor::of_slice(&[1.0_f32, 2.0, 3.0]);
    let expected = Tensor::of_slice(&[1.0_f32, 1.0]);
    assert_eq!(
        tensor_1.diff::<Tensor>(
            1,
            0,
            Some(Tensor::of_slice::<f32>(&[])),
            Some(Tensor::of_slice::<f32>(&[]))
        ),
        expected
    )
}

#[test]
fn test_clamp() {
    let tensor_1 = Tensor::of_slice(&[-1.0_f32, 1.0]);
    let expected = Tensor::of_slice(&[0.0_f32, 1.0]);
    assert_eq!(tensor_1.clamp_min(0.0_f64), expected)
}
