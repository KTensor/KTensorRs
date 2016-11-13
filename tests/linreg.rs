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
    let layers: usize = 2;
    let mut states = Vec::<Arc<k::State>>::with_capacity(2 * layers);
    let mut graph_head: Arc<k::Graph<f64>> = input_x.clone();

    let w = Arc::new(k::State::new(format!("weight_w_{}", 1), k::Vec2(2, 2)));
    let b = Arc::new(k::State::new(format!("weight_b_{}", 1), k::Vec2(1, 2)));

    let dot = Arc::new(k::op::dot::<f64>(format!("layer_{}_dot", 1), graph_head.clone(), w.clone()));
    let add = Arc::new(k::op::add_f64(format!("layer_{}_add", 1), dot.clone(), b.clone()));

    let relu = Arc::new(k::op::relu_f64(format!("layer_{}_relu", 1), add.clone()));

    graph_head = relu.clone();

    let softmax = Arc::new(k::op::softmax_f64(format!("layer_{}_softmax", 1), graph_head.clone()));
    let xentropy = Arc::new(k::cost::softmax_cross_entropy_f64(format!("layer_{}_xentropy", 1), softmax.clone(), target_y.clone()));

    states.push(w.clone());
    states.push(b.clone());

    let w2 = Arc::new(k::State::new(format!("weight_w_{}", 2), k::Vec2(2, 2)));
    let b2 = Arc::new(k::State::new(format!("weight_b_{}", 2), k::Vec2(1, 2)));

    let dot2 = Arc::new(k::op::dot::<f64>(format!("layer_{}_dot", 2), graph_head.clone(), w2.clone()));
    let add2 = Arc::new(k::op::add_f64(format!("layer_{}_add", 2), dot2.clone(), b2.clone()));

    let relu2 = Arc::new(k::op::relu_f64(format!("layer_{}_relu", 2), add2.clone()));

    states.push(w2.clone());
    states.push(b2.clone());

    graph_head = relu2.clone();


    // initialize states
    let mut state_context = k::Context::<f64>::with_capacity(2 * layers);
    k::state::init_f64(states, &mut state_context);

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

    variable_context.set(input_x.get_id(), training_set.0.clone());
    variable_context.set(target_y.get_id(), training_set.1.clone());
    println!("\ninput x");
    let final_test = k::execute(input_x.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\nweight w 1");
    let final_test = k::execute(w.clone(), &state_context, &variable_context);
    for i in 0..2 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\ndot");
    let final_test = k::execute(dot.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\nweight b 1");
    let final_test = k::execute(b.clone(), &state_context, &variable_context);
    for i in 0..1 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\nadd");
    let final_test = k::execute(add.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\nrelu");
    let final_test = k::execute(relu.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\ntarget y");
    let final_test = k::execute(target_y.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\nxentropy");
    let final_test = k::execute(xentropy.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }

    let mut history = k::Context::<f64>::with_capacity(5 * layers + 4);

    k::train(xentropy.clone(), &mut state_context, &variable_context, &mut history, -0.001);

    println!("\ninput x");
    let final_test = k::execute(input_x.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\nweight w 1");
    let final_test = k::execute(w.clone(), &state_context, &variable_context);
    for i in 0..2 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\ndot");
    let final_test = k::execute(dot.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\nweight b 1");
    let final_test = k::execute(b.clone(), &state_context, &variable_context);
    for i in 0..1 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\nadd");
    let final_test = k::execute(add.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\nrelu");
    let final_test = k::execute(relu.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\ntarget y");
    let final_test = k::execute(target_y.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
    println!("\nxentropy");
    let final_test = k::execute(xentropy.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(k::Vec2(i, 0)), final_test.get(k::Vec2(i, 1)));
    }
}
