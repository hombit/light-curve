use crate::evaluator::*;

/// Median magnitude
///
/// $$
/// \mathrm{Median}
/// $$
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
#[derive(Clone, Default, Debug)]
pub struct Median {}

lazy_info!(
    MEDIAN_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl Median {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for Median
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.m.get_median()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &MEDIAN_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["median"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["median magnitude"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(median_info, Median::default());

    feature_test!(
        median_odd,
        [Box::new(Median::new())],
        [3.0],
        [-99.0, 0.0, 3.0, 3.1, 3.2],
    );

    feature_test!(
        median_even,
        [Box::new(Median::new())],
        [1.5],
        [-99.0, 0.0, 4.0, 3.0, 2.0, 1.0],
    );
}
