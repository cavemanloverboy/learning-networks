use std::io::{Read, Write};

use fabricator::{Factory, Network};
use serde_cbor::ser::to_vec_packed;
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
    vs.load("emulator.ot").expect("failed to load emulator");

    // Get weights and biases and put them in expected order
    let lock = vs.variables_.lock().unwrap();
    let mut named_tensors: Vec<(&String, &Tensor)> = lock.named_variables.iter().collect();
    named_tensors.sort_by_key(|(name, _tensor)| name.clone());

    // Package weights and biases
    let weights_and_biases: Vec<(Vec<f32>, Vec<f32>)> = named_tensors
        .chunks_exact(2)
        .into_iter()
        .map(|chunk| {
            let [(n1, bias), (n2, weights)]: [(&String, &Tensor); 2] = chunk.try_into().unwrap();
            assert!(n1.contains("bias"));
            assert!(n2.contains("weight"));
            let weights: Vec<f32> = weights.contiguous().view(-1).into();
            let bias: Vec<f32> = bias.contiguous().view(-1).into();
            println!("weights {} bias {}", weights.len(), bias.len());
            (weights, bias)
        })
        .collect();

    // Save weights and biases to disk
    let mut bytes: Vec<u8> = to_vec_packed(&weights_and_biases).unwrap();
    let mut file = std::fs::File::create("emulator.al").expect("failed to make file");
    file.write_all(&bytes).unwrap();
    file.flush().unwrap();
    drop(file);

    // Load bytes and ensure all is well
    let mut file = std::fs::File::open("emulator.al").unwrap();
    bytes.clear();
    file.read_to_end(&mut bytes);
    let loaded_weights_and_bias: Vec<(Vec<f32>, Vec<f32>)> =
        serde_cbor::from_slice(&bytes).unwrap();
    assert_eq!(
        &weights_and_biases, &loaded_weights_and_bias,
        "something went wrong during i/o"
    );
}
