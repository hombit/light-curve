use crate::evaluator::*;

/// Mean magnitude
///
/// $$
/// \langle m \rangle \equiv \frac1{N} \sum_i m_i.
/// $$
/// This is non-weighted mean, see [WeightedMean](crate::WeightedMean) for weighted mean.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
#[derive(Clone, Default, Debug)]
pub struct Mean {}

lazy_info!(
    MEAN_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl Mean {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for Mean
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.m.get_mean()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &MEAN_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["mean"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["mean magnitude"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(mean_info, Mean::default());

    feature_test!(
        mean,
        [Mean::new()],
        [14.0],
        [1.0_f32, 1.0, 1.0, 1.0, 5.0, 6.0, 6.0, 6.0, 99.0],
    );
}
