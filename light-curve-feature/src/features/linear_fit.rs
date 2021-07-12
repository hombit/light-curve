use crate::evaluator::*;
use crate::fit::fit_straight_line;

/// The slope, its error and reduced $\chi^2$ of the light curve in the linear fit
///
/// Least squares fit of the linear stochastic model with Gaussian noise described by observation
/// errors $\\{\delta_i\\}$:
/// $$
/// m_i = c + \mathrm{slope}\\,t_i + \delta_i \varepsilon_i
/// $$
/// where $c$ is a constant,
/// $\\{\varepsilon_i\\}$ are standard distributed random variables.
///
/// Feature values are $\mathrm{slope}$, $\sigma_\mathrm{slope}$ and
/// $\frac{\sum{((m_i - c - \mathrm{slope}\\,t_i) / \delta_i)^2}}{N - 2}$.
///
/// - Depends on: **time**, **magnitude**, **magnitude error**
/// - Minimum number of observations: **3**
/// - Number of features: **3**
#[derive(Clone, Default, Debug)]
pub struct LinearFit {}

impl LinearFit {
    pub fn new() -> Self {
        Self {}
    }
}

lazy_info!(
    LINEAR_FIT_INFO,
    size: 3,
    min_ts_length: 3,
    t_required: true,
    m_required: true,
    w_required: true,
    sorting_required: true,
);

impl<T> FeatureEvaluator<T> for LinearFit
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let result = fit_straight_line(ts, true);
        Ok(vec![
            result.slope,
            T::sqrt(result.slope_sigma2),
            result.reduced_chi2,
        ])
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &LINEAR_FIT_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec![
            "linear_fit_slope",
            "linear_fit_slope_sigma",
            "linear_fit_reduced_chi2",
        ]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![
            "slope of linear fit",
            "error of slope of linear fit",
            "linear fit quality (reduced chi2)",
        ]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(linear_fit_info, LinearFit::default());

    feature_test!(
        linear_fit,
        [Box::new(LinearFit::default())],
        [1.0544186045473263, 0.7963978113902943, 0.013781209302325587],
        [0.0_f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.0_f32, 0.01, 0.04, 0.09, 0.16, 0.25, 0.36, 0.49, 0.64, 0.81, 1.0],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
    );
}
