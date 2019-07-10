#[macro_use] extern crate clap;
#[macro_use] extern crate log;
#[macro_use] extern crate failure;
extern crate tch;
extern crate pretty_env_logger;

pub mod model;
pub mod utils;

use std::path::PathBuf;
use tch::{
    IndexOp,
    Device,
    Kind,
    nn::{
        VarStore,
        ModuleT,
        Adam,
        OptimizerConfig,
    },
};
use failure::Fallible;
use crate::model::MobileNetV3;
use crate::model::Mode;

fn main() -> Fallible<()> {
    pretty_env_logger::init();
    let arg_yaml = load_yaml!("args.yaml");
    let arg_matches = clap::App::from_yaml(arg_yaml).get_matches();

    let dataset_dir = match arg_matches.value_of("dataset") {
        Some(path) => PathBuf::from(path),
        None => bail!("dataset directory is not specified"),
    };
    let n_steps = match arg_matches.value_of("n_steps") {
        Some(n_steps) => n_steps.parse()?,
        None => 1000,
    };
    let batch_size: i64 = match arg_matches.value_of("batch_size") {
        Some(bsize) => bsize.parse()?,
        None => 128,
    };
    let dropout = match arg_matches.value_of("dropout") {
        Some(dropout) => dropout.parse()?,
        None => 0.8,
    };
    let width_mult = match arg_matches.value_of("width_mult") {
        Some(width_mult) => width_mult.parse()?,
        None => 1.0,
    };

    info!("Load dataset from {:?}", &dataset_dir);
    let mnist_dataset = tch::vision::mnist::load_dir(&dataset_dir)?;
    let n_train_samples = mnist_dataset.train_images.size()[0];
    let n_test_samples = mnist_dataset.test_images.size()[0];
    let input_size: i64 = 28;
    let input_channel: i64 = 1;
    let n_classes = 10;
    info!("{} training samples. {} testing samples", n_train_samples, n_test_samples);

    assert_eq!(
        mnist_dataset.train_images.size()[1],
        input_size.pow(2) * input_channel,
    );

    let vs = VarStore::new(Device::Cuda(0));
    let root = vs.root();
    let model = MobileNetV3::new(
        &root / "mobilenetv3",
        input_channel,
        n_classes,
        dropout,
        width_mult,
        Mode::Large,
    )?;
    let opt = Adam::default().build(&vs, 1e-3)?;

    let train_iter = (0..n_train_samples)
        .into_iter()
        .step_by(batch_size as usize)
        .cycle()
        .map(|begin| (begin, (begin + batch_size).min(n_train_samples)))
        .map(|(begin, end)| {
            let samples = mnist_dataset.train_images.i(begin..end)
                .view(&[end - begin, input_channel, input_size, input_size])
                .to_device(Device::Cuda(0));
            let labels = mnist_dataset.train_labels.i(begin..end)
                .to_device(Device::Cuda(0));
            (samples, labels)
        });

    for (step, (samples, labels)) in train_iter.enumerate().take(n_steps) {
        let prediction_logits = model.forward_t(&samples, true);
        let prediction_argmax = prediction_logits.argmax(1, false);
        let precision: f64 = prediction_argmax
            .eq1(&labels)
            .to_kind(Kind::Float)
            .mean()
            .into();
        let loss = prediction_logits.cross_entropy_for_logits(&labels);
        opt.backward_step(&loss);

        let loss_val = f64::from(&loss);
        println!("step: {}\tloss: {}\tprecision: {}", step, loss_val, precision);
    }

    Ok(())
}
