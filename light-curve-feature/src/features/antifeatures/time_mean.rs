use crate::evaluator::*;

/// Mean time
///
/// $$
/// \langle t \rangle \equiv \frac1{N} \sum_i {t_i}.
/// $$
///
/// - Depends on: **time**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
#[derive(Clone, Default, Debug)]
pub struct TimeMean {}

impl TimeMean {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    TIME_MEAN_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: true,
    m_required: false,
    w_required: false,
    sorting_required: false,
);

impl<T> FeatureEvaluator<T> for TimeMean
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.t.get_mean()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &TIME_MEAN_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["ANTIFEATURE_time_mean"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["mean of time moments"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(time_mean_info, TimeMean::default());

    feature_test!(
        time_mean,
        [TimeMean::new()],
        [14.0],
        [1.0_f32, 1.0, 1.0, 1.0, 5.0, 6.0, 6.0, 6.0, 99.0],
    );
}
