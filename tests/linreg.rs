extern crate ktensor as k;

#[test]
fn linear_regression() {
    // Variables
    let input_x = k::Variable::new("input_x", k::Vec2(1, 2));
    let target_y = k::Variable::new("target_y", k::Vec2(1, 2));

    // States
    let weight_w = k::State::new("weight_w", k::Vec2(2, 2));

    // Initialize
    let mut variables = k::Context::<f64>::with_capacity(1);
    k::variable::init_f64(vec![&input_x, &target_y], &mut variables);
    let mut states = k::Context::<f64>::with_capacity(1);
    k::state::init_f64(vec![&weight_w], &mut states);

    // Graph
    let layer_1 = k::op::dot::<f64>("layer_1", &input_x, &weight_w);
    let softmax = k::op::softmax_f64("softmax", &layer_1);
    let xentropy = k::cost::softmax_cross_entropy_f64("xentropy", &softmax, &target_y);

    // Training
    let mut history = k::Context::<f64>::with_capacity(6);
    let print_rate = 4096;

    let training_set = vec![
        (k::Tensor::from_vec(k::Vec2(1, 2), vec![0.0, 0.0]), k::Tensor::from_vec(k::Vec2(1, 2), vec![0.0, 1.0])),
        (k::Tensor::from_vec(k::Vec2(1, 2), vec![0.0, 1.0]), k::Tensor::from_vec(k::Vec2(1, 2), vec![1.0, 0.0])),
        (k::Tensor::from_vec(k::Vec2(1, 2), vec![1.0, 0.0]), k::Tensor::from_vec(k::Vec2(1, 2), vec![1.0, 0.0])),
        (k::Tensor::from_vec(k::Vec2(1, 2), vec![1.0, 1.0]), k::Tensor::from_vec(k::Vec2(1, 2), vec![0.0, 1.0])),
    ];

    for i in 0..256 {
        // init batch


        k::train(xentropy, &mut states, &variables, &mut history, 0.001);
        if i % print_rate == 0 {
            println!("{}", );
        }
    }
}
