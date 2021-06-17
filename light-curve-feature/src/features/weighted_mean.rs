use crate::evaluator::*;

/// Weighted mean magnitude
///
/// $$
/// \bar{m} \equiv \frac{\sum_i m_i / \delta_i^2}{\sum_i 1 / \delta_i^2}.
/// $$
/// See [Mean](crate::Mean) for non-weighted mean.
///
/// - Depends on: **magnitude**, **magnitude error**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
#[derive(Clone, Default, Debug)]
pub struct WeightedMean {}

lazy_info!(
    WEIGHTED_MEAN_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: true,
    sorting_required: false,
);

impl WeightedMean {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for WeightedMean
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.get_m_weighted_mean()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &WEIGHTED_MEAN_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["weighted_mean"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["magnitude averaged weighted by inverse square error"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(weighted_mean_info, WeightedMean::default());

    feature_test!(
        weighted_mean,
        [Box::new(WeightedMean::new())],
        [1.1777777777777778],
        [1.0; 5], // isn't used
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
        [10.0, 5.0, 3.0, 2.5, 2.0],
    );
}
