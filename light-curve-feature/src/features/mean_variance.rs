use crate::evaluator::*;

/// Standard deviation to mean ratio
///
/// $$
/// \frac{\sigma_m}{\langle m \rangle}
/// $$
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
#[derive(Clone, Debug, Default)]
pub struct MeanVariance {}

lazy_info!(
    MEAN_VARIANCE_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl MeanVariance {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for MeanVariance
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.m.get_std() / ts.m.get_mean()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &MEAN_VARIANCE_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["mean_variance"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["standard deviation of magnitude to its mean value ratio"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(mean_variance_info, MeanVariance::default());

    feature_test!(
        mean,
        [MeanVariance::new()],
        [2.2832017440606585],
        [1.0_f32, 1.0, 1.0, 1.0, 5.0, 6.0, 6.0, 6.0, 99.0],
    );
}
