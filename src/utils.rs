use std::borrow::Borrow;
use tch::nn;
use crate::model::NL;

// Helper functions
pub fn conv_layer<'a, P: Borrow<nn::Path<'a>>>(
    path: P,
    input_channel: i64,
    output_channel: i64,
    kernel: i64,
    stride: i64,
    padding: i64,
    groups: Option<i64>,
) -> nn::Conv2D {
    let mut config = nn::ConvConfig {
        stride: stride as i64,
        padding: padding as i64,
        bias: false,
        ..Default::default()
    };

    if let Some(g) = groups {
        config.groups = g as i64;
    }

    nn::conv2d(
        path,
        input_channel as i64,
        output_channel as i64,
        kernel as i64,
        config,
    )
}

pub fn norm_layer<'a, P: Borrow<nn::Path<'a>>>(
    path: P,
    channel: i64,
) -> nn::BatchNorm
{
    nn::batch_norm2d(
        path,
        channel as i64,
        nn::BatchNormConfig {
            cudnn_enabled: true,
            ..Default::default()
        }
    )
}

pub fn nlin_layer<'a>(nl: NL) -> nn::Func<'a> {
    match nl {
        NL::ReLU => {
            nn::func(|xs| xs.relu())
        }
        NL::Hswish => {
            nn::func(|xs| xs.apply(&hswish()))
        }
    }
}

pub fn conv_1x1_bn_layer<'a, P: Borrow<nn::Path<'a>>>(
    path: P,
    input_channel: i64,
    output_channel: i64,
    nl: NL,
) -> nn::SequentialT
{
    let pathb = path.borrow();
    let conv = conv_layer(
        pathb / "conv",
        input_channel,
        output_channel,
        1,
        1,
        0,
        None,
    );
    let bn = norm_layer(
        pathb / "bn",
        output_channel,
    );

    nn::seq_t()
        .add(conv)
        .add(bn)
        .add(nlin_layer(nl))
}

pub fn hswish<'a>() -> nn::Func<'a> {
    nn::func(|xs| {
        xs * (xs + 3.)
            .relu()
            .clamp_max_(6.) / 6.
    })
}
