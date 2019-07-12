#[macro_use] extern crate clap;
#[macro_use] extern crate failure;
extern crate tch;

pub mod model;
pub mod utils;

use std::path::PathBuf;
use tch::{
    IndexOp,
    Tensor,
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

enum DatasetName {
    MNIST,
    Cifar10,
}

type PreprocessorType = Box<dyn Fn(&Tensor, &Tensor) -> (Tensor, Tensor)>;

fn main() -> Fallible<()> {
    // Parse args
    let arg_yaml = load_yaml!("args.yaml");
    let arg_matches = clap::App::from_yaml(arg_yaml).get_matches();

    let dataset_name = match arg_matches.value_of("dataset_name") {
        Some(name) => match name {
            "mnist" => DatasetName::MNIST,
            "cifar-10" => DatasetName::Cifar10,
            _ => bail!("dataset name {:?} is not supported", name),
        },
        None => bail!("dataset name is not specified"),
    };
    let dataset_dir = match arg_matches.value_of("dataset_dir") {
        Some(path) => PathBuf::from(path),
        None => bail!("dataset directory is not specified"),
    };
    let model_file = match arg_matches.value_of("model_file") {
        Some(path) => Some(PathBuf::from(path)),
        None => None,
    };
    let epochs = match arg_matches.value_of("epochs") {
        Some(n_steps) => n_steps.parse()?,
        None => 10,
    };
    let batch_size: i64 = match arg_matches.value_of("batch_size") {
        Some(bsize) => bsize.parse()?,
        None => 32,
    };
    let learning_rate: f64 = match arg_matches.value_of("learning_rate") {
        Some(lr) => lr.parse()?,
        None => 1e-3,
    };
    let dropout = match arg_matches.value_of("dropout") {
        Some(dropout) => dropout.parse()?,
        None => 0.8,
    };
    let width_mult = match arg_matches.value_of("width_mult") {
        Some(width_mult) => width_mult.parse()?,
        None => 1.0,
    };

    // Load dataset
    println!("Load dataset from {:?}", &dataset_dir);

    let (dataset, input_size, input_channel, n_classes, preprocessor) = match dataset_name {
        DatasetName::MNIST => {
            let dataset = tch::vision::mnist::load_dir(&dataset_dir)?;
            let input_size: i64 = 28;
            let input_channel: i64 = 1;
            let n_classes = 10;
            let preprocessor: PreprocessorType = Box::new(
                move |images, labels| {
                    let reshaped_images =  images.view(&[-1, input_channel, input_size, input_size]);
                    (reshaped_images, labels.shallow_clone())
                }
            );
            (dataset, input_size, input_channel, n_classes, preprocessor)
        }
        DatasetName::Cifar10 => {
            let dataset = tch::vision::cifar::load_dir(&dataset_dir)?;
            let input_size: i64 = 32;
            let input_channel: i64 = 3;
            let n_classes = 10;
            let preprocessor: PreprocessorType = Box::new(
                move |images, labels| {
                    (images.shallow_clone(), labels.shallow_clone())
                }
            );
            (dataset, input_size, input_channel, n_classes, preprocessor)
        }
    };
    let n_train_samples = dataset.train_images.size()[0];
    let n_test_samples = dataset.test_images.size()[0];
    println!("{} training samples, {} testing samples, {} classes", n_train_samples, n_test_samples, n_classes);

    // Initialize model and optimizer
    let mut vs = VarStore::new(Device::Cuda(0));
    let root = vs.root();
    let model = MobileNetV3::new(
        &root / "mobilenetv3",
        input_channel,
        n_classes,
        dropout,
        width_mult,
        Mode::Large,
    )?;
    let opt = Adam::default().build(&vs, learning_rate)?;

    // Try to load model parameters
    if let Some(path) = &model_file {
        println!("Load model parameters from {:?}", path);
        vs.load(path)?;
    }

    // Train model
    let train_iter = (0..n_train_samples)
        .into_iter()
        .step_by(batch_size as usize)
        .cycle()
        .scan(
            0,
            |epoch, begin| {
                if begin == 0 { *epoch += 1; }
                if *epoch <= epochs {
                    Some((*epoch, begin, (begin + batch_size).min(n_train_samples)))
                }
                else {
                    None
                }
            }
        )
        .map(|(epoch, begin, end)| {
            let images = dataset.train_images.i(begin..end)
                .view(&[end - begin, input_channel, input_size, input_size])
                .to_device(Device::Cuda(0));
            let labels = dataset.train_labels.i(begin..end)
                .to_device(Device::Cuda(0));
            let (images, labels) = preprocessor(&images, &labels);
            (epoch, images, labels)
        })
        .enumerate();
        // .take(n_train_samples as usize * epochs);

    for (step, (epoch, images, labels)) in train_iter {
        let prediction_logits = model.forward_t(&images, true);
        let prediction_argmax = prediction_logits.argmax(1, false);
        let precision: f64 = prediction_argmax
            .eq1(&labels)
            .to_kind(Kind::Float)
            .mean()
            .into();
        let loss = prediction_logits.cross_entropy_for_logits(&labels);
        opt.backward_step(&loss);

        let loss_val = f64::from(&loss);
        println!("epoch: {:6},\tstep: {:6},\tloss: {:.3},\tprecision: {:.3}", epoch, step, loss_val, precision);
    }

    // Try to store model parameters
    if let Some(path) = &model_file {
        println!("Save model parameters to {:?}", path);
        vs.save(path)?;
    }

    // Inference on test dataset
    let total_count = (0..n_test_samples)
        .into_iter()
        .step_by(batch_size as usize)
        .map(|begin| (begin, (begin + batch_size).min(n_test_samples)))
        .map(|(begin, end)| {
            let images = dataset.test_images.i(begin..end)
                .view(&[end - begin, input_channel, input_size, input_size])
                .to_device(Device::Cuda(0));
            let labels = dataset.test_labels.i(begin..end)
                .to_device(Device::Cuda(0));
            let (images, labels) = preprocessor(&images, &labels);
            (begin, end, images, labels)
        })
        .fold(
            0_i64,
            |prev_count, (begin, end, images, labels)| {
                let prediction_logits = model.forward_t(&images, false);
                let prediction_argmax = prediction_logits.argmax(1, false);
                let prediction_vec = Vec::<i64>::from(&prediction_argmax);
                let expect_vec = Vec::<i64>::from(&labels);
                let is_correct = Vec::<i64>::from(prediction_argmax.eq1(&labels));
                let correct_count: i64 = is_correct.iter().sum();

                let iter = (begin..end)
                    .zip(prediction_vec.into_iter())
                    .zip(expect_vec.into_iter())
                    .zip(is_correct.into_iter());

                for (((idx, predict), expect), is_correct) in iter {
                    let comment = match is_correct == 1 {
                        true => "correct",
                        false => "wrong",
                    };
                    println!("index: {:6},\texpect: {},\tprediction: {},\tcomment: {}", idx, expect, predict, comment);
                }

                prev_count + correct_count
            }
        );

    let precision = total_count as f64 / n_test_samples as f64;
    println!("{} out of {} predictions are correct on test dataset (precision: {})", total_count, n_test_samples, precision);

    Ok(())
}
