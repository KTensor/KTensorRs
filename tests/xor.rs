extern crate ktensor as k;
use k::{Arc, Vec2, Tensor, Context, Graph, State, Variable};


#[test]
#[ignore]
fn xor() {
    ///////////////
    // Variables //
    ///////////////

    let input_x = Arc::new(Variable::new("input_x".to_string(), Vec2(0, 2)));
    let target_y = Arc::new(Variable::new("target_y".to_string(), Vec2(0, 2)));

    // Initialize
    let mut variable_context = Context::<f64>::with_capacity(2);
    Variable::init_f64(vec![input_x.clone(), target_y.clone()], &mut variable_context);


    ///////////
    // Graph //
    ///////////

    let layers: usize = 2;
    let mut states = Vec::<Arc<State>>::with_capacity(2 * layers);
    let mut graph_head: Arc<Graph<f64>> = input_x.clone();

    let w = Arc::new(State::new(format!("weight_w_{}", 1), Vec2(2, 4)));
    let b = Arc::new(State::new(format!("weight_b_{}", 1), Vec2(1, 4)));

    let dot = Arc::new(k::op::dot::<f64>(format!("layer_{}_dot", 1), graph_head.clone(), w.clone()));
    let add = Arc::new(k::op::add::<f64>(format!("layer_{}_add", 1), dot.clone(), b.clone()));

    let relu = Arc::new(k::op::relu_f64(format!("layer_{}_relu", 1), add.clone()));

    graph_head = relu.clone();

    states.push(w.clone());
    states.push(b.clone());

    let w2 = Arc::new(State::new(format!("weight_w_{}", 2), Vec2(4, 2)));
    let b2 = Arc::new(State::new(format!("weight_b_{}", 2), Vec2(1, 2)));

    let dot2 = Arc::new(k::op::dot::<f64>(format!("layer_{}_dot", 2), graph_head.clone(), w2.clone()));
    let add2 = Arc::new(k::op::add::<f64>(format!("layer_{}_add", 2), dot2.clone(), b2.clone()));

    let relu2 = Arc::new(k::op::relu_f64(format!("layer_{}_relu", 2), add2.clone()));

    graph_head = relu2.clone();

    states.push(w2.clone());
    states.push(b2.clone());

    let softmax2 = Arc::new(k::op::softmax_f64(format!("layer_{}_softmax", 2), graph_head.clone()));
    let xentropy2 = Arc::new(k::cost::softmax_cross_entropy_f64(format!("layer_{}_xentropy", 2), softmax2.clone(), target_y.clone()));

    // initialize states
    let mut state_context = Context::<f64>::with_capacity(2 * layers);
    State::init_f64(states, &mut state_context);

    //////////////
    // training //
    //////////////

    let training_set = (
        Tensor::from_vec(Vec2(4, 2), vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
        ]),
        Tensor::from_vec(Vec2(4, 2), vec![
        0.0, 1.0,
        1.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
        ]),
    );

    let training_vec = vec![
        (Tensor::from_vec(Vec2(1, 2), vec![0.0, 0.0,]), Tensor::from_vec(Vec2(1, 2), vec![0.0, 1.0,])),
        (Tensor::from_vec(Vec2(1, 2), vec![0.0, 1.0,]), Tensor::from_vec(Vec2(1, 2), vec![1.0, 0.0,])),
        (Tensor::from_vec(Vec2(1, 2), vec![1.0, 0.0,]), Tensor::from_vec(Vec2(1, 2), vec![1.0, 0.0,])),
        (Tensor::from_vec(Vec2(1, 2), vec![1.0, 1.0,]), Tensor::from_vec(Vec2(1, 2), vec![0.0, 1.0,])),
    ];

    let batch = true;
    let learning_rate = -0.1;
    let print_rate = 4096;

    let mut history = Context::<f64>::with_capacity(5 * layers + 4);

    if batch {
        variable_context.set(input_x.get_id(), training_set.0.clone());
        variable_context.set(target_y.get_id(), training_set.1.clone());
    }
    for i in 0..16384 {
        let (ref a, ref b) = training_vec[i % 4];
        if !batch {
            variable_context.set(input_x.get_id(), a.clone());
            variable_context.set(target_y.get_id(), b.clone());
        }
        k::train(xentropy2.clone(), &mut state_context, &variable_context, &mut history, learning_rate);

        // test print
        if i % print_rate == 0 {
            variable_context.set(input_x.get_id(), training_set.0.clone());
            variable_context.set(target_y.get_id(), training_set.1.clone());
            println!("\niteration: {} | xentropy2 output:", i);
            let final_test = k::execute(xentropy2.clone(), &state_context, &variable_context);
            for i in 0..4 {
                println!("{} {}", final_test.get(Vec2(i, 0)), final_test.get(Vec2(i, 1)));
            }
        }
    }

    ////////////////
    // final test //
    ////////////////

    variable_context.set(input_x.get_id(), training_set.0.clone());
    variable_context.set(target_y.get_id(), training_set.1.clone());
    println!("\nfinal test | xentropy2 output:");
    let final_test = k::execute(xentropy2.clone(), &state_context, &variable_context);
    for i in 0..4 {
        println!("{} {}", final_test.get(Vec2(i, 0)), final_test.get(Vec2(i, 1)));
    }
}
