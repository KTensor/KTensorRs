extern crate ktensor as k;
use k::{Arc, Vec2, Tensor, Context, Graph, State, Variable};
use std::path::{Path};
use std::fs::{File};
use std::io::{BufReader, Read};
use std::error::{Error};
use std::cmp;

fn read_u8(reader: &mut Read) -> u8 {
    use std::mem;

    let mut buf: [u8; 1] = [0];
    match reader.read_exact(&mut buf).map(|_| {
        let data: u8 = unsafe { mem::transmute(buf) };
        data
    }) {
        Err(reason) => panic!("failed to read u8: {}", Error::description(&reason)),
        Ok(byte)    => byte,
    }
}

fn read_u32(reader: &mut Read) -> u32 {
    use std::mem;

    let mut buf: [u8; 4] = [0, 0, 0, 0];
    match reader.read_exact(&mut buf).map(|_| {
        let data: u32 = unsafe { mem::transmute(buf) };
        data
    }) {
        Err(reason) => panic!("failed to read u32: {}", Error::description(&reason)),
        Ok(byte)    => byte,
    }
}

fn read_mnist(labels_path: &Path, labels_checknum: u32, data_path: &Path, data_checknum: u32, batch_size: usize, samples: Option<usize>) {
    let labels_file = match File::open(&labels_path) {
        Err(reason) => panic!("failed to open {}: {}", labels_path.display(), Error::description(&reason)),
        Ok(file)    => file,
    };
    let data_file = match File::open(&data_path) {
        Err(reason) => panic!("failed to open {}: {}", data_path.display(), Error::description(&reason)),
        Ok(file)    => file,
    };
    let ref mut labels_reader = BufReader::new(labels_file);
    let ref mut data_reader = BufReader::new(data_file);

    let labels_magic = u32::from_be(read_u32(labels_reader));
    let data_magic = u32::from_be(read_u32(data_reader));

    assert_eq!(labels_magic, labels_checknum);
    assert_eq!(data_magic, data_checknum);

    let labels_count = u32::from_be(read_u32(labels_reader)) as usize;
    let data_count = u32::from_be(read_u32(data_reader)) as usize;
    let sample_count = cmp::min(labels_count, data_count);
    let sample_count = cmp::min(sample_count, samples.unwrap_or(sample_count));

    let rows = u32::from_be(read_u32(data_reader)) as usize;
    let columns = u32::from_be(read_u32(data_reader)) as usize;

    let mut sample_vec: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(sample_count/batch_size);
    for _ in 0..sample_count/batch_size {
        let mut sample_data = vec![0u8; batch_size*rows*columns];
        match data_reader.read_exact(sample_data.as_mut()) {
            Err(reason) => panic!("failed to read data byte array: {}", Error::description(&reason)),
            Ok(_)       => (),
        }
        let mut sample_labels = vec![0u8; batch_size];
        match labels_reader.read_exact(sample_labels.as_mut()) {
            Err(reason) => panic!("failed to read labels byte array: {}", Error::description(&reason)),
            Ok(_)       => (),
        }
        sample_vec.push((sample_data, sample_labels));
    }
}

#[test]
#[ignore]
fn mnist(){
    ///////////////
    // Variables //
    ///////////////

    let input_x = Arc::new(Variable::new("input_x".to_string(), Vec2(0, 28 * 28)));
    let target_y = Arc::new(Variable::new("target_y".to_string(), Vec2(0, 10)));

    // Initialize
    let mut variable_context = Context::<f32>::with_capacity(2);
    Variable::init_f32(vec![input_x.clone(), target_y.clone()], &mut variable_context);


    ///////////
    // Graph //
    ///////////

    let layers: usize = 2;
    let mut states = Vec::<Arc<State>>::with_capacity(2 * layers);
    let mut graph_head: Arc<Graph<f32>> = input_x.clone();

    {
        let w = Arc::new(State::new(format!("weight_w_{}", 1), Vec2(28 * 28, 16)));
        let b = Arc::new(State::new(format!("weight_b_{}", 1), Vec2(1, 16)));

        let dot = Arc::new(k::op::dot::<f32>(format!("layer_{}_dot", 1), graph_head.clone(), w.clone()));
        let add = Arc::new(k::op::add::<f32>(format!("layer_{}_add", 1), dot, b.clone()));

        let relu = Arc::new(k::op::relu_f32(format!("layer_{}_relu", 1), add));

        graph_head = relu;

        states.push(w);
        states.push(b);
    }

    {
        let w = Arc::new(State::new(format!("weight_w_{}", 2), Vec2(16, 10)));
        let b = Arc::new(State::new(format!("weight_b_{}", 2), Vec2(1, 10)));

        let dot = Arc::new(k::op::dot::<f32>(format!("layer_{}_dot", 2), graph_head.clone(), w.clone()));
        let add = Arc::new(k::op::add::<f32>(format!("layer_{}_add", 2), dot, b.clone()));

        let relu = Arc::new(k::op::relu_f32(format!("layer_{}_relu", 2), add));

        graph_head = relu;

        states.push(w);
        states.push(b);
    }

    let softmax = Arc::new(k::op::softmax_f32(format!("layer_{}_softmax", 3), graph_head.clone()));
    let xentropy = Arc::new(k::cost::softmax_cross_entropy_f32(format!("layer_{}_xentropy", 3), softmax.clone(), target_y.clone()));

    // initialize states
    let mut state_context = Context::<f32>::with_capacity(2 * layers);
    State::init_f32(states, &mut state_context);


    //////////////
    // Training //
    //////////////

    let iterations = 16384;
    let learning_rate = -0.1;
    let print_rate = 4096;

    let train_labels_path = Path::new("data/train-labels-idx1-ubyte");
    let train_data_path = Path::new("data/train-images-idx3-ubyte");
    read_mnist(&train_labels_path, 2049, &train_data_path, 2051, 16, Some(4096));


    let training_vec = vec![(
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
    )];


    let mut history = Context::<f32>::with_capacity(5 * layers + 4);

    for i in 0..iterations {
        let (ref a, ref b) = training_vec[i % training_vec.len()];
        variable_context.set(input_x.get_id(), a.clone());
        variable_context.set(target_y.get_id(), b.clone());

        k::train(xentropy.clone(), &mut state_context, &variable_context, &mut history, learning_rate);

        // test print
        if i % print_rate == 0 {
            println!("\niteration: {} | cross entropy cost: {}", i, k::execute(xentropy.clone(), &state_context, &variable_context).get(Vec2(0, 0)));
        }
    }

    ////////////////
    // Final test //
    ////////////////

}
