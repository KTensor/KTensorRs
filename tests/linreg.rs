extern crate ktensor as k;

fn relu_layer_f64(layer_number: usize, x: &k::Graph<f64>, w: &k::Graph<f64>, b: &k::Graph<f64>) -> (k::Graph<f64>, k::Graph<f64>, k::Graph<f64>) {
    let dot = k::op::dot::<f64>(format!("layer_{}_dot", layer_number), x, w);
    let add = k::op::add::<f64>(format!("layer_{}_add", layer_number), &dot, b);
    (dot, add, k::op::relu_f64(format!("layer_{}_relu", layer_number), &add))
}

#[test]
fn linear_regression() {
    // Variables
    let input_x = k::Variable::new("input_x".to_string(), k::Vec2(1, 2));
    let target_y = k::Variable::new("target_y".to_string(), k::Vec2(1, 2));

    // Initialize
    let mut variables = k::Context::<f64>::with_capacity(2);
    k::variable::init_f64(vec![&input_x, &target_y], &mut variables);

    // States
    let weight_w = k::State::new("weight_w".to_string(), k::Vec2(2, 2));
    let weight_b = k::State::new("weight_b".to_string(), k::Vec2(1, 2));


    // Graph
    let mut states = Vec::<k::State>::with_capacity(2 * 2);
    let mut graphs = Vec::<k::Graph<f64>>::with_capacity(3 * 2);
    let mut graph_head: &k::Graph<f64> = &input_x;

    for i in 0..2 {
        let w = k::State::new(format!("weight_w_{}", i), k::Vec2(2, 2));
        let b = k::State::new(format!("weight_b_{}", i), k::Vec2(1, 2));
        let (dot, add, relu) = relu_layer_f64(i, graph_head, &w, &b);
        graph_head = &relu;
        graphs.push(dot);
        graphs.push(add);
        graphs.push(relu);
        states.push(w);
        states.push(b);
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


        k::train(&xentropy, &mut states, &variables, &mut history, &0.001);
        if i % print_rate == 0 {

        }
    }
}
