use crate::evaluator::*;

/// Cusum — a range of cumulative sums
///
/// $$
/// \mathrm{cusum} \equiv \max(S) - \min(S),
/// $$
/// where
/// $$
/// S_j \equiv \frac1{N\\,\sigma_m} \sum_{i=0}^j{\left(m\_i - \langle m \rangle\right)},
/// $$
/// $N$ is the number of observations,
/// $\langle m \rangle$ is the mean magnitude
/// and $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// Kim et al. 2014, [DOI:10.1051/0004-6361/201323252](https://doi.org/10.1051/0004-6361/201323252)
#[derive(Clone, Default, Debug)]
pub struct Cusum {}

impl Cusum {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    CUSUM_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: true,
);

impl<T> FeatureEvaluator<T> for Cusum
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_std = get_nonzero_m_std(ts)?;
        let m_mean = ts.m.get_mean();
        let (_last_cusum, min_cusum, max_cusum) = ts.m.sample.iter().fold(
            (T::zero(), T::infinity(), -T::infinity()),
            |(mut cusum, min_cusum, max_cusum), &m| {
                cusum += m - m_mean;
                (cusum, T::min(min_cusum, cusum), T::max(max_cusum, cusum))
            },
        );
        Ok(vec![(max_cusum - min_cusum) / (m_std * ts.lenf())])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &CUSUM_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["cusum"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["range of cumulative sums of magnitudes"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(cusum_info, Cusum::default());

    feature_test!(
        cumsum,
        [Box::new(Cusum::new())],
        [0.3589213],
        [1.0_f32, 1.0, 1.0, 5.0, 8.0, 20.0],
    );
}
