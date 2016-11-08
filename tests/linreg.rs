extern crate ktensor as k;

#[test]
fn linear_regression() {
    let x = k::Variable::new("input_x", k::Vec2(1, 2));
    let mut variables = k::Context::from_vec(vec![
        (&x, k::Tensor::from_vec(k::Vec2(1, 2), vec![0.0, 0.0])),
    ]);

    let w = k::State::new("weight_w", k::Vec2(2, 1));
    let mut states = k::Context::<f64>::with_capacity(1);
    k::init_state_f64(vec![&w], &mut states);

    k::op::dot::<f64>("layer_1", &x, &w);
}
