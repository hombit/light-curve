use crate::periodogram::FftwFloat;

use conv::prelude::*;
use lazy_static::lazy_static;
use ndarray::Array0;
use num_traits::{float::Float as NumFloat, float::FloatConst, FromPrimitive};
use std::cmp::PartialOrd;
use std::fmt::{Debug, Display, LowerExp};
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign};

lazy_static! {
    static ref ARRAY0_UNITY_F32: Array0<f32> = Array0::from_elem((), 1.0);
}

lazy_static! {
    static ref ARRAY0_UNITY_F64: Array0<f64> = Array0::from_elem((), 1.0);
}

pub trait Float:
    'static
    + NumFloat
    + FloatConst
    + FromPrimitive
    + PartialOrd
    + Sum
    + ValueFrom<u32>
    + ValueFrom<usize>
    + ValueFrom<f32>
    + ValueInto<f64>
    + ApproxFrom<usize>
    + ApproxFrom<f64>
    + ApproxInto<usize, RoundToNearest>
    + ApproxInto<f64>
    + Clone
    + Copy
    + Send
    + Sync
    + AddAssign
    + MulAssign
    + DivAssign
    + Display
    + Debug
    + LowerExp
    + FftwFloat
{
    fn half() -> Self;
    fn two() -> Self;
    fn three() -> Self;
    fn four() -> Self;
    fn five() -> Self;
    fn ten() -> Self;
    fn hundred() -> Self;
    fn array0_unity() -> &'static Array0<Self>;
}

impl Float for f32 {
    #[inline]
    fn half() -> Self {
        0.5
    }

    #[inline]
    fn two() -> Self {
        2.0
    }

    #[inline]
    fn three() -> Self {
        3.0
    }

    #[inline]
    fn four() -> Self {
        4.0
    }

    #[inline]
    fn five() -> Self {
        5.0
    }

    #[inline]
    fn ten() -> Self {
        10.0
    }

    #[inline]
    fn hundred() -> Self {
        100.0
    }

    fn array0_unity() -> &'static Array0<Self> {
        &ARRAY0_UNITY_F32
    }
}

impl Float for f64 {
    #[inline]
    fn half() -> Self {
        0.5
    }

    #[inline]
    fn two() -> Self {
        2.0
    }

    #[inline]
    fn three() -> Self {
        3.0
    }

    #[inline]
    fn four() -> Self {
        4.0
    }

    #[inline]
    fn five() -> Self {
        5.0
    }

    #[inline]
    fn ten() -> Self {
        10.0
    }

    #[inline]
    fn hundred() -> Self {
        100.0
    }

    fn array0_unity() -> &'static Array0<Self> {
        &ARRAY0_UNITY_F64
    }
}
