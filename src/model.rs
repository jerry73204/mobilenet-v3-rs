use std::borrow::Borrow;
use failure::Fallible;
use tch::{
    Tensor,
    nn::{
        self,
        Module,
        ModuleT,
        Sequential,
        SequentialT,
    }
};
use crate::utils;

#[derive(Debug)]
pub struct MobileNetV3 {
    features: SequentialT,
    classifier: SequentialT,
}

#[derive(Debug)]
pub struct MobileBottleneck {
    use_res_connect: bool,
    conv_block: SequentialT,
}

#[derive(Debug)]
pub struct SEModule {
    fc: Sequential,
}

#[derive(Debug, Copy, Clone)]
pub enum Mode {
    Small, Large
}

#[derive(Debug, Copy, Clone)]
pub enum NL {
    ReLU, Hswish,
}

#[derive(Debug, Copy, Clone)]
pub enum SE {
    SEModule, Identity,
}

impl MobileNetV3 {
    pub fn new<'a, P: Borrow<nn::Path<'a>>>(
        path: P,
        input_channel: i64,
        n_classes: i64,
        dropout: f64,
        width_mult: f64,
        mode: Mode
    ) -> Fallible<MobileNetV3>
    {
        ensure!(input_channel.is_positive(), "input_channel must be positive integer, but get {}", input_channel);
        ensure!(n_classes.is_positive(), "n_classes must be positive integer, but get {}", n_classes);
        ensure!(!dropout.is_sign_negative(), "dropout should be non-negative, but get {}", dropout);
        ensure!(width_mult.is_sign_positive(), "width_mult should be positive, but get {}", width_mult);

        let default_last_channel = 1280;
        let pathb = path.borrow();

        let mobile_setting = match mode {
            Mode::Small => {
                vec![
                    // k, exp, c, se,           nl,         s,
                    (3, 16,  16,  SE::SEModule, NL::ReLU,   2),
                    (3, 72,  24,  SE::Identity, NL::ReLU,   2),
                    (3, 88,  24,  SE::Identity, NL::ReLU,   1),
                    (5, 96,  40,  SE::SEModule, NL::Hswish, 2),
                    (5, 240, 40,  SE::SEModule, NL::Hswish, 1),
                    (5, 240, 40,  SE::SEModule, NL::Hswish, 1),
                    (5, 120, 48,  SE::SEModule, NL::Hswish, 1),
                    (5, 144, 48,  SE::SEModule, NL::Hswish, 1),
                    (5, 288, 96,  SE::SEModule, NL::Hswish, 2),
                    (5, 576, 96,  SE::SEModule, NL::Hswish, 1),
                    (5, 576, 96,  SE::SEModule, NL::Hswish, 1),
                ]
            }
            Mode::Large => {
                vec![
                    // k, exp, c, se,           nl,         s,
                    (3, 16,  16,  SE::Identity, NL::ReLU,   1),
                    (3, 64,  24,  SE::Identity, NL::ReLU,   2),
                    (3, 72,  24,  SE::Identity, NL::ReLU,   1),
                    (5, 72,  40,  SE::SEModule, NL::ReLU,   2),
                    (5, 120, 40,  SE::SEModule, NL::ReLU,   1),
                    (5, 120, 40,  SE::SEModule, NL::ReLU,   1),
                    (3, 240, 80,  SE::Identity, NL::Hswish, 2),
                    (3, 200, 80,  SE::Identity, NL::Hswish, 1),
                    (3, 184, 80,  SE::Identity, NL::Hswish, 1),
                    (3, 184, 80,  SE::Identity, NL::Hswish, 1),
                    (3, 480, 112, SE::SEModule, NL::Hswish, 1),
                    (3, 672, 112, SE::SEModule, NL::Hswish, 1),
                    (5, 672, 160, SE::SEModule, NL::Hswish, 2),
                    (5, 960, 160, SE::SEModule, NL::Hswish, 1),
                    (5, 960, 160, SE::SEModule, NL::Hswish, 1),
                ]
            }
        };

        // Helper functions
        let make_divisible = |val| {
            let divisible_by = 8.;
            let rounded = (val as f64 / divisible_by).ceil() * divisible_by;
            rounded as i64
        };

        // Compute last channel
        let last_channel = if width_mult > 1. {
            make_divisible(default_last_channel as f64 * width_mult)
        }
        else {
            default_last_channel
        };

        // Build mobile blocks
        let mut features = nn::seq_t();
        let mut c_in = input_channel;

        for (ind, (k, exp, c, se, nl, s)) in mobile_setting.into_iter().enumerate()
        {
            let c_out = make_divisible(c as f64 * width_mult);
            let c_exp = make_divisible(exp as f64 * width_mult);
            let block = MobileBottleneck::new(
                pathb / format!("bottleneck_{}", ind),
                c_in,
                c_out,
                k,
                s,
                c_exp,
                se,
                nl,
            )?;

            features = features.add(block);
            c_in = c_out;
        }

        // Build last several layers
        let last_conv_channel = match mode {
            Mode::Large => make_divisible(960. * width_mult),
            Mode::Small => make_divisible(576. * width_mult),
        };

        let conv_1x1_bn = utils::conv_1x1_bn_layer(
            pathb / "conv_1x1_bn",
            c_in,
            last_conv_channel,
            NL::Hswish,
        );
        let conv_last = utils::conv_layer(
            pathb / "conv_last",
            last_conv_channel,
            last_channel,
            1,
            1,
            0,
            None,
        );

        features = features.add(conv_1x1_bn)
            .add_fn(|xs| xs.adaptive_avg_pool2d(&[1, 1]))
            .add(conv_last)
            .add(utils::hswish());

        // Build classifier
        let classifier_linear = nn::linear(
            pathb / "classifier_linear",
            last_channel as i64,
            n_classes as i64,
            nn::LinearConfig {
                ..Default::default()
            },
        );

        let classifier = nn::seq_t()
            .add_fn_t(move |xs, train| xs.dropout(dropout, train))
            .add(classifier_linear);


        let model = MobileNetV3 {
            features,
            classifier,
        };

        Ok(model)
    }
}

impl ModuleT for MobileNetV3 {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let height = xs.size()[2];
        let width = xs.size()[3];
        assert!(height == width, "Input tensor should be square in size");
        // assert!(height % 32 == 0, "Input tensor height/width should be multiple of 32");

        xs.apply_t(&self.features, train)
            .mean2(&[2, 3], false)
            .apply_t(&self.classifier, train)
    }
}

impl MobileBottleneck {
    pub fn new<'a, P: Borrow<nn::Path<'a>>>(
        path: P,
        input_channel: i64,
        output_channel: i64,
        kernel: i64,
        stride: i64,
        exp_channel: i64,
        se: SE,
        nl: NL,
    ) -> Fallible<MobileBottleneck>
    {
        ensure!(vec![1, 2].contains(&stride), "stride should be either 1 or 2");
        ensure!(vec![3, 5].contains(&kernel), "kernel should be either 3 or 5");

        let pathb = path.borrow();
        let padding = (kernel - 1) / 2;
        let use_res_connect = stride == 1 && input_channel == output_channel;

        let se_module = SEModule::new(pathb / "se_module", exp_channel, None);
        let se_layer = move |xs: &Tensor| {
            match se {
                SE::SEModule => {
                    xs.apply(&se_module)
                }
                SE::Identity => {
                    xs.shallow_clone()
                }
            }
        };

        // pw
        let conv_pw = utils::conv_layer(
            pathb / "conv_pw",
            input_channel,
            exp_channel,
            1,
            1,
            0,
            None,
        );
        let bn_pw = utils::norm_layer(
            pathb / "bn_pw",
            exp_channel,
        );

        // dw
        let conv_dw = utils::conv_layer(
            pathb / "conv_dw",
            exp_channel,
            exp_channel,
            kernel,
            stride,
            padding,
            Some(exp_channel),
        );
        let bn_dw = utils::norm_layer(
            pathb / "bn_dw",
            exp_channel,
        );

        // pw-linear
        let conv_pw_linear = utils::conv_layer(
            pathb / "conv_pw_linear",
            exp_channel,
            output_channel,
            1,
            1,
            0,
            None,
        );
        let bn_pw_linear = utils::norm_layer(
            pathb / "bn_pw_linear",
            output_channel,
        );

        // Build conv block
        let conv_block = nn::seq_t()
            .add(conv_pw)
            .add(bn_pw)
            .add(utils::nlin_layer(nl))
            .add(conv_dw)
            .add(bn_dw)
            .add_fn(se_layer)
            .add(utils::nlin_layer(nl))
            .add(conv_pw_linear)
            .add(bn_pw_linear);

        let block = MobileBottleneck {
            use_res_connect,
            conv_block,
        };
        Ok(block)
    }
}

impl ModuleT for MobileBottleneck {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        match self.use_res_connect {
            true => xs.apply_t(&self.conv_block, train) + xs,
            false => xs.apply_t(&self.conv_block, train),
        }
    }
}

impl SEModule {
    fn new<'a, P: Borrow<nn::Path<'a>>>(
        path: P,
        channel: i64,
        reduction: Option<i64>,
    ) -> SEModule
    {
        let pathb = path.borrow();
        let reduction_ = match reduction {
            Some(r) => r,
            None => 4,
        };

        let linear_layer_1 = nn::linear(
            pathb / "linear1",
            channel as i64,
            (channel / reduction_) as i64,
            nn::LinearConfig {
                bs_init: None,
                ..Default::default()
            },
        );

        let linear_layer_2 = nn::linear(
            pathb / "linear2",
            (channel / reduction_) as i64,
            channel as i64,
            nn::LinearConfig {
                bs_init: None,
                ..Default::default()
            },
        );

        let fc = nn::seq()
            .add(linear_layer_1)
            .add_fn(|xs| xs.relu())
            .add(linear_layer_2)
            .add_fn(|xs| {      // HSigmoid function
                (xs + 3.).relu().clamp_max_(6.) / 6.
            });

        SEModule {
            fc,
        }
    }
}

impl Module for SEModule {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let size = xs.size();
        let bsize = size[0];
        let channel = size[1];
        let y = xs.adaptive_avg_pool2d(&[1, 1])
            .view([bsize, channel])
            .apply(&self.fc)
            .view([bsize, channel, 1, 1]);
        xs * y.expand_as(&xs)
    }
}
