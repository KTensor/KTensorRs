extern crate ktensor as k;
use std::sync::{Arc};

#[test]
fn linear_regression() {
    // Variables
    let input_x = Arc::new(k::Variable::new("input_x".to_string(), k::Vec2(0, 2)));
    let target_y = Arc::new(k::Variable::new("target_y".to_string(), k::Vec2(0, 2)));

    // Initialize
    let mut variable_context = k::Context::<f64>::with_capacity(2);
    k::variable::init_f64(vec![input_x.clone(), target_y.clone()], &mut variable_context);


    // Graph
    let layers: usize = 1;
    let mut states = Vec::<Arc<k::State>>::with_capacity(2 * layers);
    let mut graph_head: Arc<k::Graph<f64>> = input_x.clone();

    {
        let w = Arc::new(k::State::new(format!("weight_w_{}", 1), k::Vec2(2, 2)));
        let b = Arc::new(k::State::new(format!("weight_b_{}", 1), k::Vec2(1, 2)));

        let dot = Arc::new(k::op::dot::<f64>(format!("layer_{}_dot", 1), graph_head, w.clone()));
        let add = Arc::new(k::op::add::<f64>(format!("layer_{}_add", 1), dot, b.clone()));
        let relu = Arc::new(k::op::relu_f64(format!("layer_{}_relu", 1), add));

        states.push(w);
        states.push(b);
        graph_head = relu;
    }

    // initialize states
    let mut state_context = k::Context::<f64>::with_capacity(2 * layers);
    k::state::init_f64(states, &mut state_context);

    let softmax = Arc::new(k::op::softmax_f64("softmax".to_string(), graph_head));
    let xentropy = Arc::new(k::cost::softmax_cross_entropy_f64("xentropy".to_string(), softmax, target_y.clone()));

    // Training
    let mut history = k::Context::<f64>::with_capacity(5 * layers + 2 + 2);
    let print_rate = 4096;

    let training_set = (
        k::Tensor::from_vec(k::Vec2(4, 2), vec![
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0,
        ]),
        k::Tensor::from_vec(k::Vec2(4, 2), vec![
            0.0, 1.0,
            1.0, 0.0,
            1.0, 0.0,
            0.0, 1.0,
        ]),
    );

    for i in 0..256 {
        // init batch
        variable_context.set(input_x.get_id(), training_set.0.clone());
        variable_context.set(target_y.get_id(), training_set.1.clone());

        k::train(xentropy.clone(), &mut state_context, &variable_context, &mut history, -0.001);
        if i % print_rate == 0 {

        }
    }
}
