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
    let train_labels_path = Path::new("data/train-labels-idx1-ubyte");
    let train_data_path = Path::new("data/train-images-idx3-ubyte");
    read_mnist(&train_labels_path, 2049, &train_data_path, 2051, 16, Some(4096));


    ///////////////
    // Variables //
    ///////////////

    let input_x = Arc::new(Variable::new("input_x".to_string(), Vec2(0, 28 * 28)));
    let target_y = Arc::new(Variable::new("target_y".to_string(), Vec2(0, 10)));

    ///////////
    // Graph //
    ///////////

    let layers: usize = 2;
    let mut states = Vec::<Arc<State>>::with_capacity(2 * layers);
    let mut graph_head: Arc<Graph<f64>> = input_x.clone();

    {
        let w = Arc::new(State::new(format!("weight_w_{}", 1), Vec2(28 * 28, 16)));
        let b = Arc::new(State::new(format!("weight_b_{}", 1), Vec2(1, 16)));

        let dot = Arc::new(k::op::dot::<f64>(format!("layer_{}_dot", 1), graph_head.clone(), w.clone()));
        let add = Arc::new(k::op::add::<f64>(format!("layer_{}_add", 1), dot.clone(), b.clone()));

        let relu = Arc::new(k::op::relu_f64(format!("layer_{}_relu", 1), add.clone()));

        graph_head = relu.clone();

        states.push(w.clone());
        states.push(b.clone());
    }

    {
        let w = Arc::new(State::new(format!("weight_w_{}", 1), Vec2(2, 4)));
        let b = Arc::new(State::new(format!("weight_b_{}", 1), Vec2(1, 4)));

        let dot = Arc::new(k::op::dot::<f64>(format!("layer_{}_dot", 1), graph_head.clone(), w.clone()));
        let add = Arc::new(k::op::add::<f64>(format!("layer_{}_add", 1), dot.clone(), b.clone()));

        let relu = Arc::new(k::op::relu_f64(format!("layer_{}_relu", 1), add.clone()));

        graph_head = relu.clone();

        states.push(w.clone());
        states.push(b.clone());
    }
}
