extern crate tch;
#[macro_use] extern crate failure;

mod model;
mod utils;

use tch::{Device, nn::VarStore};
use failure::Error;
use crate::model::MobileNetV3;
use crate::model::Mode;

fn main() -> Result<(), Error> {
    // TODO parse arguments

    let vs = VarStore::new(Device::Cuda(0));
    let path = vs.root();
    let model = MobileNetV3::new(
        &path,
        1000,
        224,
        0.8,
        1.0,
        Mode::Small,
    )?;

    // TODO load train data

    Ok(())
}
