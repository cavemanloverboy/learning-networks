use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Linear {
    ws: Matrix,
    bs: Vector,
}

impl Linear {
    pub fn new(ws: Matrix, bs: Vector) -> Linear {
        assert_eq!(
            ws.out_dim * ws.in_dim,
            ws.data.len(),
            "ws width * height != length"
        );
        assert_eq!(
            ws.out_dim,
            bs.len(),
            "weight out_dim and bias length do not match"
        );
        Linear { ws, bs }
    }
    pub fn forward_batch(&self, xs: &Vec<Vector>) -> Vec<Vector> {
        let mut outputs = vec![vec![0.0; self.ws.out_dim]; xs.len()];
        for (i, x) in xs.into_iter().enumerate() {
            let output: &mut Vec<f32> = outputs.get_mut(i).unwrap();
            self.ws.mx_b_inplace(&x, &self.bs, output);
        }
        outputs
    }
    pub fn forward(&self, x: &[f32]) -> Vector {
        let mut output = vec![0.0; self.ws.out_dim];
        self.ws.mx_b_inplace(&x, &self.bs, &mut output);
        output
    }
    pub fn forward_relu(&self, x: &[f32]) -> Vector {
        let mut output = vec![0.0; self.ws.out_dim];
        self.ws.mx_b_inplace(&x, &self.bs, &mut output);
        for element in &mut output {
            if *element < 0.0 {
                *element = 0.0;
            }
        }
        output
    }
    pub fn forward_batch_relu(&self, xs: &Vec<Vector>) -> Vec<Vector> {
        let mut outputs = vec![vec![0.0; self.ws.out_dim]; xs.len()];
        for (i, x) in xs.into_iter().enumerate() {
            let output: &mut Vec<f32> = outputs.get_mut(i).unwrap();
            self.ws.mx_b_inplace(&x, &self.bs, output);
        }
        for output_vec in &mut outputs {
            for element in output_vec {
                if *element < 0.0 {
                    *element = 0.0;
                }
            }
        }
        outputs
    }
}

#[derive(Serialize, Deserialize)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

pub type Vector = Vec<f32>;

pub trait Methods {
    fn relu(self) -> Self;
}

impl Matrix {
    fn mx_b_inplace(&self, x: &[f32], b: &Vector, output: &mut Vec<f32>) {
        assert_eq!(self.in_dim, x.len());
        assert_eq!(self.out_dim, b.len());

        for (i, col) in self.data.chunks_exact(self.in_dim).enumerate() {
            let mut acc = 0.0;
            for (idx, element) in col.into_iter().enumerate() {
                acc += element * x[idx];
            }
            output[i] = acc + b[i];
        }
    }
}

#[test]
fn test_mx_b() {
    // Inputs
    let xs: Vec<Vector> = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![4.0, 8.0]];

    // Weights and biases
    let ws: Matrix = Matrix {
        data: vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        in_dim: 2,
        out_dim: 3,
    };
    let bs: Vector = vec![1.0, 2.0, 3.0];

    // Initialize linear
    let linear: Linear = Linear::new(ws, bs);

    let output = linear.forward_batch(&xs);

    assert_eq!(
        output,
        vec![
            vec![10.0, 14.0, 18.0],
            vec![19.0, 26.0, 33.0],
            vec![37.0, 50.0, 63.0]
        ]
    )
}

#[test]
fn test_mx_b_double() {
    // Inputs
    let xs: Vec<Vector> = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![4.0, 8.0]];

    // Weights and biases
    let ws: Matrix = Matrix {
        data: vec![1.0, 4.0, 2.0, 5.0],
        in_dim: 2,
        out_dim: 2,
    };
    let bs: Vector = vec![1.0, 2.0];

    // Initialize linear
    let linear: Linear = Linear::new(ws, bs);

    let output = linear.forward_batch(&xs);
    assert_eq!(
        *output,
        vec![vec![10.0, 14.0], vec![19.0, 26.0,], vec![37.0, 50.0]]
    );
    let output = linear.forward_batch(&output);
    assert_eq!(
        *output,
        vec![vec![67.0, 92.0], vec![124.0, 170.0], vec![238.0, 326.0]]
    )
}

#[test]
fn test_matmul_relu() {
    // Inputs
    let xs: Vec<Vector> = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![4.0, 8.0]];

    // Weights and biases
    let ws: Matrix = Matrix {
        data: vec![1.0, 4.0, 2.0, -5.0],
        in_dim: 2,
        out_dim: 2,
    };
    let bs: Vector = vec![1.0, 2.0];

    // Initialize linear
    let linear: Linear = Linear::new(ws, bs);

    let output = linear.forward_batch(&xs);
    assert_eq!(
        *output,
        vec![vec![10.0, -6.0], vec![19.0, -14.0,], vec![37.0, -30.0]]
    );
    let output = linear.forward_batch_relu(&xs);
    assert_eq!(
        *output,
        vec![vec![10.0, 0.0], vec![19.0, 0.0], vec![37.0, 0.0]]
    );
}
