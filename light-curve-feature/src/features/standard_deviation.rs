use crate::evaluator::*;

/// Standard deviation of magnitude $\sigma_m$
///
/// $$
/// \sigma_m \equiv \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)},
/// $$
///
/// $N$ is the number of observations
/// and $\langle m \rangle$ is the mean magnitude.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Standard_deviation)
#[derive(Clone, Default, Debug)]
pub struct StandardDeviation {}

lazy_info!(
    STANDARD_DEVIATION_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl StandardDeviation {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for StandardDeviation
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.m.get_std()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &STANDARD_DEVIATION_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["standard_deviation"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["standard deviation of magnitude sample"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(standard_deviation_info, StandardDeviation::default());

    feature_test!(
        standard_deviation,
        [StandardDeviation::new()],
        [1.5811388300841898],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
    );
}
