use crate::float_trait::Float;
use conv::*;
use std::fmt::Debug;

pub trait ErrorFunction<T>: Clone + Debug
where
    T: ErfFloat,
{
    fn erf(x: T) -> T;

    fn normal_cdf(x: T, mean: T, sigma: T) -> T {
        T::half() * (T::one() + Self::erf((x - mean) / sigma * T::FRAC_1_SQRT_2()))
    }

    fn max_dx_nonunity_normal_cdf(sigma: T) -> T;

    fn min_dx_nonzero_normal_cdf(sigma: T) -> T {
        -Self::max_dx_nonunity_normal_cdf(sigma)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ExactErf;

impl<T> ErrorFunction<T> for ExactErf
where
    T: ErfFloat,
{
    fn erf(x: T) -> T {
        x.libm_erf()
    }

    fn max_dx_nonunity_normal_cdf(sigma: T) -> T {
        T::SQRT_2_ERFINV_UNITY_MINUS_EPS * sigma
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Eps1Over1e3Erf;

impl<T> ErrorFunction<T> for Eps1Over1e3Erf
where
    T: ErfFloat,
{
    fn erf(x: T) -> T {
        x.erf_eps_1over1e3()
    }

    fn max_dx_nonunity_normal_cdf(sigma: T) -> T {
        T::SQRT_2_MAX_X_FOR_ERF_EPS_1OVER1E3 * sigma
    }
}

pub trait ErfFloat: Float + ApproxInto<usize, RoundToZero> + num_traits::Float {
    const SQRT_2_ERFINV_UNITY_MINUS_EPS: Self;

    fn libm_erf(self) -> Self;

    const SQRT_2_MAX_X_FOR_ERF_EPS_1OVER1E3: Self;
    const X_FOR_ERF_EPS_1OVER1E3: [Self; 64];
    const INVERSED_DX_FOR_ERF_EPS_1OVER1E3: Self;
    const Y_FOR_ERF_EPS_1OVER1E3: [Self; 64];
    fn erf_eps_1over1e3(self) -> Self {
        match self {
            _ if self < Self::X_FOR_ERF_EPS_1OVER1E3[0] => -Self::one(),
            _ if self >= Self::X_FOR_ERF_EPS_1OVER1E3[63] => Self::one(),
            x => {
                let idx =
                    (x - Self::X_FOR_ERF_EPS_1OVER1E3[0]) * Self::INVERSED_DX_FOR_ERF_EPS_1OVER1E3;
                let alpha = idx.fract();
                let idx: usize = idx.approx_by::<RoundToZero>().unwrap();
                Self::Y_FOR_ERF_EPS_1OVER1E3[idx] * (Self::one() - alpha)
                    + Self::Y_FOR_ERF_EPS_1OVER1E3[idx + 1] * alpha
            }
        }
    }
}

#[allow(clippy::excessive_precision)]
impl ErfFloat for f32 {
    const SQRT_2_ERFINV_UNITY_MINUS_EPS: Self = 5.294704084854598;

    fn libm_erf(self) -> Self {
        libm::erff(self)
    }

    const SQRT_2_MAX_X_FOR_ERF_EPS_1OVER1E3: Self = 3.389783571270326;
    const X_FOR_ERF_EPS_1OVER1E3: [Self; 64] = [
        -2.39693895,
        -2.32084565,
        -2.24475235,
        -2.16865905,
        -2.09256575,
        -2.01647245,
        -1.94037915,
        -1.86428585,
        -1.78819255,
        -1.71209925,
        -1.63600595,
        -1.55991265,
        -1.48381935,
        -1.40772605,
        -1.33163275,
        -1.25553945,
        -1.17944615,
        -1.10335285,
        -1.02725955,
        -0.95116625,
        -0.87507295,
        -0.79897965,
        -0.72288635,
        -0.64679305,
        -0.57069975,
        -0.49460645,
        -0.41851315,
        -0.34241985,
        -0.26632655,
        -0.19023325,
        -0.11413995,
        -0.03804665,
        0.03804665,
        0.11413995,
        0.19023325,
        0.26632655,
        0.34241985,
        0.41851315,
        0.49460645,
        0.57069975,
        0.64679305,
        0.72288635,
        0.79897965,
        0.87507295,
        0.95116625,
        1.02725955,
        1.10335285,
        1.17944615,
        1.25553945,
        1.33163275,
        1.40772605,
        1.48381935,
        1.55991265,
        1.63600595,
        1.71209925,
        1.78819255,
        1.86428585,
        1.94037915,
        2.01647245,
        2.09256575,
        2.16865905,
        2.24475235,
        2.32084565,
        2.39693895,
    ];
    const INVERSED_DX_FOR_ERF_EPS_1OVER1E3: Self = 13.141761468984605;
    const Y_FOR_ERF_EPS_1OVER1E3: [Self; 64] = [
        -0.99930052,
        -0.99896989,
        -0.99849936,
        -0.99783743,
        -0.99691696,
        -0.9956517,
        -0.99393249,
        -0.99162334,
        -0.98855749,
        -0.98453378,
        -0.97931372,
        -0.97261948,
        -0.96413348,
        -0.9534999,
        -0.94032851,
        -0.92420128,
        -0.90468204,
        -0.88132908,
        -0.85371082,
        -0.82142392,
        -0.78411334,
        -0.74149338,
        -0.69336849,
        -0.6396527,
        -0.58038613,
        -0.51574736,
        -0.44606033,
        -0.37179495,
        -0.29356079,
        -0.21209374,
        -0.12823602,
        -0.04291034,
        0.04291034,
        0.12823602,
        0.21209374,
        0.29356079,
        0.37179495,
        0.44606033,
        0.51574736,
        0.58038613,
        0.6396527,
        0.69336849,
        0.74149338,
        0.78411334,
        0.82142392,
        0.85371082,
        0.88132908,
        0.90468204,
        0.92420128,
        0.94032851,
        0.9534999,
        0.96413348,
        0.97261948,
        0.97931372,
        0.98453378,
        0.98855749,
        0.99162334,
        0.99393249,
        0.9956517,
        0.99691696,
        0.99783743,
        0.99849936,
        0.99896989,
        0.99930052,
    ];
}

impl ErfFloat for f64 {
    const SQRT_2_ERFINV_UNITY_MINUS_EPS: Self = 8.20953615160139;

    fn libm_erf(self) -> Self {
        libm::erf(self)
    }

    const SQRT_2_MAX_X_FOR_ERF_EPS_1OVER1E3: Self = 3.389783571270326;
    const X_FOR_ERF_EPS_1OVER1E3: [Self; 64] = [
        -2.39693895,
        -2.32084565,
        -2.24475235,
        -2.16865905,
        -2.09256575,
        -2.01647245,
        -1.94037915,
        -1.86428585,
        -1.78819255,
        -1.71209925,
        -1.63600595,
        -1.55991265,
        -1.48381935,
        -1.40772605,
        -1.33163275,
        -1.25553945,
        -1.17944615,
        -1.10335285,
        -1.02725955,
        -0.95116625,
        -0.87507295,
        -0.79897965,
        -0.72288635,
        -0.64679305,
        -0.57069975,
        -0.49460645,
        -0.41851315,
        -0.34241985,
        -0.26632655,
        -0.19023325,
        -0.11413995,
        -0.03804665,
        0.03804665,
        0.11413995,
        0.19023325,
        0.26632655,
        0.34241985,
        0.41851315,
        0.49460645,
        0.57069975,
        0.64679305,
        0.72288635,
        0.79897965,
        0.87507295,
        0.95116625,
        1.02725955,
        1.10335285,
        1.17944615,
        1.25553945,
        1.33163275,
        1.40772605,
        1.48381935,
        1.55991265,
        1.63600595,
        1.71209925,
        1.78819255,
        1.86428585,
        1.94037915,
        2.01647245,
        2.09256575,
        2.16865905,
        2.24475235,
        2.32084565,
        2.39693895,
    ];
    const INVERSED_DX_FOR_ERF_EPS_1OVER1E3: Self = 13.141761468984605;
    const Y_FOR_ERF_EPS_1OVER1E3: [Self; 64] = [
        -0.99930052,
        -0.99896989,
        -0.99849936,
        -0.99783743,
        -0.99691696,
        -0.9956517,
        -0.99393249,
        -0.99162334,
        -0.98855749,
        -0.98453378,
        -0.97931372,
        -0.97261948,
        -0.96413348,
        -0.9534999,
        -0.94032851,
        -0.92420128,
        -0.90468204,
        -0.88132908,
        -0.85371082,
        -0.82142392,
        -0.78411334,
        -0.74149338,
        -0.69336849,
        -0.6396527,
        -0.58038613,
        -0.51574736,
        -0.44606033,
        -0.37179495,
        -0.29356079,
        -0.21209374,
        -0.12823602,
        -0.04291034,
        0.04291034,
        0.12823602,
        0.21209374,
        0.29356079,
        0.37179495,
        0.44606033,
        0.51574736,
        0.58038613,
        0.6396527,
        0.69336849,
        0.74149338,
        0.78411334,
        0.82142392,
        0.85371082,
        0.88132908,
        0.90468204,
        0.92420128,
        0.94032851,
        0.9534999,
        0.96413348,
        0.97261948,
        0.97931372,
        0.98453378,
        0.98855749,
        0.99162334,
        0.99393249,
        0.9956517,
        0.99691696,
        0.99783743,
        0.99849936,
        0.99896989,
        0.99930052,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    #[test]
    fn erf_eps_1over1e3() {
        let x = Array1::linspace(-5.0, 5.0, 1 << 20);
        let desired = x.mapv(f32::libm_erf);
        let actual = x.mapv(f32::erf_eps_1over1e3);
        assert_abs_diff_eq!(
            actual.as_slice().unwrap(),
            desired.as_slice().unwrap(),
            epsilon = 7e-4,
        );
    }
}
