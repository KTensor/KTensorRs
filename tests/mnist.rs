extern crate ktensor as k;
use std::path::{Path};
use std::fs::{File};
use std::io::{self, BufReader, Read};
use std::error::{Error};
use std::cmp::{min};

fn read_u8(reader: &mut Read) -> u8 {
    use std::mem;

    let mut buf: [u8; 1] = [0];
    match reader.read_exact(&mut buf).map(|_| {
        let data: u8 = unsafe { mem::transmute(buf) };
        data
    }) {
        Err(reason) => panic!("failed to read u8: {}", Error::description(&reason)),
        Ok(file)    => file,
    }
}

fn read_u32(reader: &mut Read) -> u32 {
    use std::mem;

    let mut buf: [u8; 4] = [0, 0, 0, 0];
    match reader.read_exact(&mut buf).map(|_| {
        let data: u32 = unsafe { mem::transmute(buf) };
        data
    }) {
        Err(reason) => panic!("failed to read u8: {}", Error::description(&reason)),
        Ok(file)    => file,
    }
}

fn read_mnist(labels_path: &Path, labels_checknum: u32, data_path: &Path, data_checknum: u32, samples: usize) {
    let mut labels_file = match File::open(&labels_path) {
        Err(reason) => panic!("failed to open {}: {}", labels_path.display(), Error::description(&reason)),
        Ok(file)    => file,
    };
    let mut data_file = match File::open(&data_path) {
        Err(reason) => panic!("failed to open {}: {}", data_path.display(), Error::description(&reason)),
        Ok(file)    => file,
    };
    let mut labels_reader = BufReader::new(labels_file);
    let mut data_reader = BufReader::new(data_file);
    let labels_magic = u32::from_be(read_u32(&mut labels_reader));
    let data_magic = u32::from_be(read_u32(&mut data_reader));
    println!("{} {}", labels_magic, data_magic);
    assert_eq!(labels_magic, labels_checknum);
    assert_eq!(data_magic, data_checknum);
}

#[test]
#[ignore]
fn mnist(){
    let train_labels_path = Path::new("data/train-labels-idx1-ubyte");
    let train_data_path = Path::new("data/train-images-idx3-ubyte");
    read_mnist(&train_labels_path, 2049, &train_data_path, 2051, 1);
}
