extern crate ktensor as k;

#[test]
fn linear_regression() {
    // Variables
    let input_x = k::Variable::new("input_x".to_string(), k::Vec2(1, 2));
    let target_y = k::Variable::new("target_y".to_string(), k::Vec2(1, 2));

    // Initialize
    let mut variable_context = k::Context::<f64>::with_capacity(2);
    k::variable::init_f64(vec![&input_x, &target_y], &mut variable_context);

    // States
    let weight_w = k::State::new("weight_w".to_string(), k::Vec2(2, 2));
    let weight_b = k::State::new("weight_b".to_string(), k::Vec2(1, 2));


    // Graph
    let layers: usize = 2;
    let mut states = Vec::<k::State>::with_capacity(2 * layers);
    let mut graphs = Vec::<k::Node<f64>>::with_capacity(3 * layers);
    let mut graph_head: &k::Graph<f64> = &input_x;

    for i in 0..layers {
        let s = states.len();
        let g = graphs.len();

        states.push(k::State::new(format!("weight_w_{}", i), k::Vec2(2, 2)));
        states.push(k::State::new(format!("weight_b_{}", i), k::Vec2(1, 2)));

        graphs.push(k::op::dot::<f64>(format!("layer_{}_dot", i), graph_head, &states[s + 0]));
        graphs.push(k::op::add::<f64>(format!("layer_{}_add", i), &graphs[g + 0], &states[s + 1]));
        graphs.push(k::op::relu_f64(format!("layer_{}_relu", i), &graphs[g + 1]));

        graph_head = &graphs[g + 2];
    }

    let mut state_context = k::Context::<f64>::with_capacity(2);
    k::state::init_f64(&states, &mut state_context);
    let softmax = k::op::softmax_f64("softmax".to_string(), graph_head);
    let xentropy = k::cost::softmax_cross_entropy_f64("xentropy".to_string(), &softmax, &target_y);

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


        k::train(&xentropy, &mut state_context, &variable_context, &mut history, &0.001);
        if i % print_rate == 0 {

        }
    }
}
