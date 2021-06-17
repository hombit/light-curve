use crate::evaluator::*;

use ndarray::Zip;

/// Excess kurtosis of magnitude
///
/// $$
/// G_2 \equiv \frac{N\\,(N + 1)}{(N - 1)(N - 2)(N - 3)} \frac{\sum_i(m_i - \langle m \rangle)^4}{\sigma_m^4}
/// \- 3\frac{(N - 1)^2}{(N - 2)(N - 3)},
/// $$
/// where $N$ is the number of observations,
/// $\langle m \rangle$ is the mean magnitude,
/// $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **4**
/// - Number of features: **1**
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Kurtosis#Estimators_of_population_kurtosis)
#[derive(Clone, Default, Debug)]
pub struct Kurtosis {}

impl Kurtosis {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    KURTOSIS_INFO,
    size: 1,
    min_ts_length: 4,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl<T> FeatureEvaluator<T> for Kurtosis
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_std2 = get_nonzero_m_std2(ts)?;
        let m_mean = ts.m.get_mean();
        let n = ts.lenf();
        let n1 = n + T::one();
        let n_1 = n - T::one();
        let n_2 = n - T::two();
        let n_3 = n - T::three();
        let forth_moment =
            Zip::from(ts.m.sample).fold(T::zero(), |sum, &m| sum + (m - m_mean).powi(4));
        let value = forth_moment / m_std2.powi(2) * n * n1 / (n_1 * n_2 * n_3)
            - T::three() * n_1.powi(2) / (n_2 * n_3);
        Ok(vec![value])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &KURTOSIS_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["kurtosis"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["unbiased excess kurtosis of magnitudes"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(kurtosis_info, Kurtosis::default());

    feature_test!(
        kurtosis,
        [Box::new(Kurtosis::new())],
        [-1.2],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
    );
}
