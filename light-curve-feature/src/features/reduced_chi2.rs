use crate::evaluator::*;

/// Reduced $\chi^2$ of magnitude measurements
///
/// $$
/// \mathrm{reduced~}\chi^2 \equiv \frac1{N-1} \sum_i\left(\frac{m_i - \bar{m}}{\delta\_i}\right)^2,
/// $$
/// where $N$ is the number of observations,
/// and $\bar{m}$ is the weighted mean magnitude.
///
/// - Depends on: **magnitude**, **magnitude error**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// [Wikipedia](https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic)
#[derive(Clone, Default, Debug)]
pub struct ReducedChi2 {}

lazy_info!(
    REDUCED_CHI2_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: true,
    sorting_required: false,
);

impl ReducedChi2 {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for ReducedChi2
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.get_m_reduced_chi2()])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &REDUCED_CHI2_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["chi2"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["reduced chi2 as a goodness of constant fit with respect to observation errors"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(reduced_chi2_info, ReducedChi2::default());

    feature_test!(
        reduced_chi2,
        [Box::new(ReducedChi2::default())],
        [2.192592592592593],
        [0.0_f64; 10], // isn't used
        [1.0, 2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, -2.0, 0.0],
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    );
}
