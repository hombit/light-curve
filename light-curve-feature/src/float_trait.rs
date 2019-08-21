use std::cmp::PartialOrd;
use std::fmt::Display;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign};

use conv::prelude::*;
use num_traits::{float::Float as NumFloat, float::FloatConst};
use serde::{Deserialize, Serialize};

pub trait Float:
    NumFloat
    + FloatConst
    + PartialOrd
    + Sum
    + ValueFrom<usize>
    + ValueFrom<f32>
    + Clone
    + AddAssign
    + MulAssign
    + DivAssign
    + Default
    + Display
    + Sync
    + Send
    + Serialize
    + for<'de> Deserialize<'de>
{
    fn half() -> Self;
    fn two() -> Self;
    fn three() -> Self;
    fn four() -> Self;
    fn five() -> Self;
    fn ten() -> Self;
    fn hundred() -> Self;
}

impl Float for f32 {
    fn half() -> Self {
        0.5
    }
    fn two() -> Self {
        2.0
    }
    fn three() -> Self {
        3.0
    }
    fn four() -> Self {
        4.0
    }
    fn five() -> Self {
        5.0
    }
    fn ten() -> Self {
        10.0
    }
    fn hundred() -> Self {
        100.0
    }
}

impl Float for f64 {
    fn half() -> Self {
        0.5
    }
    fn two() -> Self {
        2.0
    }
    fn three() -> Self {
        3.0
    }
    fn four() -> Self {
        4.0
    }
    fn five() -> Self {
        5.0
    }
    fn ten() -> Self {
        10.0
    }
    fn hundred() -> Self {
        100.0
    }
}
