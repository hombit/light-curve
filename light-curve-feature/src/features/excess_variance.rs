use crate::evaluator::*;

use ndarray::Zip;

/// Measure of the variability amplitude
///
/// $$
/// \frac{\sigma_m^2 - \langle \delta^2 \rangle}{\langle m \rangle^2},
/// $$
/// where $\langle \delta^2 \rangle$ is the mean of squared error, $\sigma_m$ is the magnitude
/// standard deviation. Note that this definition differs from
/// [Sánchez et al. 2017](https://doi.org/10.3847/1538-4357/aa9188)
///
/// - Depends on: **magnitude**, **error**
/// - Minimum number of observations: **2**
/// - Number of features: **1**
///
/// Sánchez et al. 2017 [DOI:10.3847/1538-4357/aa9188](https://doi.org/10.3847/1538-4357/aa9188)
#[derive(Clone, Default, Debug)]
pub struct ExcessVariance {}

lazy_info!(
    EXCESS_VARIANCE_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: true,
    sorting_required: false,
);

impl ExcessVariance {
    pub fn new() -> Self {
        Self {}
    }
}

impl<T> FeatureEvaluator<T> for ExcessVariance
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let mean_error2 =
            Zip::from(ts.w.sample).fold(T::zero(), |sum, w| sum + w.recip()) / ts.lenf();
        Ok(vec![
            (ts.m.get_std2() - mean_error2) / ts.m.get_mean().powi(2),
        ])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &EXCESS_VARIANCE_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec!["excess_variance"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["variability amplitude (excess of magnitude variability over typical error)"]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(excess_variance, ExcessVariance::default());

    feature_test!(
        mean,
        [Box::new(ExcessVariance::new())],
        [0.41846885813148793],
        [0.0; 9],
        [1.0_f32, 1.0, 1.0, 1.0, 5.0, 6.0, 6.0, 6.0, 7.0],
        [1.0, 0.5, 1.0, 2.0, 0.5, 2.0, 1.0, 1.0, 0.5],
    );
}
