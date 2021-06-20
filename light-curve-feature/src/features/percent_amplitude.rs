use crate::evaluator::*;

/// Maximum deviation of magnitude from its median
///
/// $$
/// \mathrm{percent~amplitude} \equiv \max_i\left|m_i - \mathrm{Median}(m)\right|
///     = \max\\{\max(m) - \mathrm{Median}(m), \mathrm{Median}(m) - \min(m)\\}.
/// $$
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
#[derive(Clone, Default, Debug)]
pub struct PercentAmplitude {}

lazy_info!(
    PERCENT_AMPLITUDE_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl PercentAmplitude {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for PercentAmplitude
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_min = ts.m.get_min();
        let m_max = ts.m.get_max();
        let m_median = ts.m.get_median();
        Ok(vec![T::max(m_max - m_median, m_median - m_min)])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &PERCENT_AMPLITUDE_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["percent_amplitude"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["maximum absolute deviation of magnitude from its median"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(percent_amplitude_info, PercentAmplitude::default());

    feature_test!(
        percent_amplitude,
        [PercentAmplitude::new()],
        [96.0],
        [1.0_f32, 1.0, 1.0, 2.0, 4.0, 5.0, 5.0, 98.0, 100.0],
    );
}
