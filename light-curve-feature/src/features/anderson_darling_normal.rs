use crate::evaluator::*;
use crate::lnerfc::ln_erfc;

use conv::ConvUtil;

/// Unbiased Anderson–Darling normality test statistic
///
/// $$
/// A^2 \equiv \left(1 + \frac4{N} - \frac{25}{N^2}\right) \left(-N - \frac1{N} \sum_{i=0}^{N-1} {(2i + 1)\ln\Phi_i + (2(N - i) - 1)\ln(1 - \Phi_i)}\right),
/// $$
/// where $\Phi_i \equiv \Phi((m_i - \langle m \rangle) / \sigma_m)$ is the cumulative distribution
/// function of the standard normal distribution,
/// $N$ is the number of observations,
/// $\langle m \rangle$ is the mean magnitude
/// and $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **4**
/// - Number of features: **1**
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Anderson–Darling_test)
#[derive(Clone, Default, Debug)]
pub struct AndersonDarlingNormal {}

impl AndersonDarlingNormal {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    ANDERSON_DARLING_NORMAL_INFO,
    size: 1,
    min_ts_length: 4,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl<T> FeatureEvaluator<T> for AndersonDarlingNormal
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let size = self.check_ts_length(ts)?;
        let m_std = get_nonzero_m_std(ts)?;
        let m_mean = ts.m.get_mean();
        let sum: f64 =
            ts.m.get_sorted()
                .as_ref()
                .iter()
                .enumerate()
                // ln \Phi(x) = -ln2 + ln_erfc(-x / sqrt2)
                // ln (1 - \Phi(x)) = -ln2 + ln_erfc(x / sqrt2)
                .map(|(i, &m)| {
                    let x = ((m - m_mean) / m_std).value_as::<f64>().unwrap()
                        * std::f64::consts::FRAC_1_SQRT_2;
                    ((2 * i + 1) as f64) * ln_erfc(-x) + ((2 * (size - i) - 1) as f64) * ln_erfc(x)
                })
                .sum();
        let n = ts.lenf();
        Ok(vec![
            (T::one() + T::four() / n - (T::five() / n).powi(2))
                * (n * (T::two() * T::LN_2() - T::one()) - sum.approx_as::<T>().unwrap() / n),
        ])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &ANDERSON_DARLING_NORMAL_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["anderson_darling_normal"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["unbiased Anderson-Darling normality test statistics"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(
        anderson_darling_normal_info,
        AndersonDarlingNormal::default()
    );

    feature_test!(
        anderson_darling_normal,
        [Box::new(AndersonDarlingNormal::new())],
        // import numpy as np
        // from scipy.stats import anderson
        // a = np.linspace(0.0, 1.0, 101)
        // anderson(a).statistic * (1.0 + 4.0/a.size - 25.0/a.size**2)
        [1.1354353876265415],
        {
            let mut m = linspace(0.0, 1.0, 101);
            let mut rng = StdRng::seed_from_u64(0);
            m.shuffle(&mut rng);
            m
        },
    );
}
