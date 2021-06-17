use crate::evaluator::*;

/// Magnitude percentage ratio
///
/// $$
/// \mathrm{magnitude~}q\mathrm{~to~}n\mathrm{~ratio} \equiv \frac{Q(1-n) - Q(n)}{Q(1-d) - Q(d)},
/// $$
/// where $n$ and $d$ denotes user defined percentage, $Q$ is the quantile function of magnitude
/// distribution.
///
/// - Depends on: **magnitude**
/// - Minimum number of observations: **1**
/// - Number of features: **1**
///
/// D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
#[derive(Clone, Debug)]
pub struct MagnitudePercentageRatio {
    quantile_numerator: f32,
    quantile_denominator: f32,
    name: String,
    description: String,
}

lazy_info!(
    MAGNITUDE_PERCENTAGE_RATIO_INFO,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl MagnitudePercentageRatio {
    pub fn new(quantile_numerator: f32, quantile_denominator: f32) -> Self {
        assert!(
            (quantile_numerator > 0.0)
                && (quantile_numerator < 0.5)
                && (quantile_denominator > 0.0)
                && (quantile_denominator < 0.5),
            "quantiles should be between zero and half"
        );
        Self {
            quantile_numerator,
            quantile_denominator,
            name: format!(
                "magnitude_percentage_ratio_{:.0}_{:.0}",
                100.0 * quantile_numerator,
                100.0 * quantile_denominator
            ),
            description: format!(
                "ratio of {:.3e}% - {:.3e}% and {:.3e}% - {:.3e}% percentile ranges of magnitude \
                sample",
                100.0 * quantile_numerator,
                100.0 * (1.0 - quantile_numerator),
                100.0 * quantile_denominator,
                100.0 * (1.0 - quantile_denominator),
            ),
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    #[inline]
    pub fn default_quantile_numerator() -> f32 {
        0.4
    }

    #[inline]
    pub fn default_quantile_denominator() -> f32 {
        0.05
    }
}
impl Default for MagnitudePercentageRatio {
    fn default() -> Self {
        Self::new(
            Self::default_quantile_numerator(),
            Self::default_quantile_denominator(),
        )
    }
}

impl<T> FeatureEvaluator<T> for MagnitudePercentageRatio
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_sorted = ts.m.get_sorted();
        let numerator =
            m_sorted.ppf(1.0 - self.quantile_numerator) - m_sorted.ppf(self.quantile_numerator);
        let denumerator =
            m_sorted.ppf(1.0 - self.quantile_denominator) - m_sorted.ppf(self.quantile_denominator);
        if numerator.is_zero() & denumerator.is_zero() {
            Err(EvaluatorError::FlatTimeSeries)
        } else {
            Ok(vec![numerator / denumerator])
        }
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &MAGNITUDE_PERCENTAGE_RATIO_INFO
    }

    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![self.description.as_str()]
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    eval_info_test!(
        magnitude_percentage_ratio_info,
        MagnitudePercentageRatio::default()
    );

    feature_test!(
        magnitude_percentage_ratio,
        [
            Box::new(MagnitudePercentageRatio::default()),
            Box::new(MagnitudePercentageRatio::new(0.4, 0.05)), // should be the same
            Box::new(MagnitudePercentageRatio::new(0.2, 0.05)),
            Box::new(MagnitudePercentageRatio::new(0.4, 0.1)),
        ],
        [0.12886598, 0.12886598, 0.7628866, 0.13586957],
        [
            80.0_f32, 13.0, 20.0, 20.0, 75.0, 25.0, 100.0, 1.0, 2.0, 3.0, 7.0, 30.0, 5.0, 9.0,
            10.0, 70.0, 80.0, 92.0, 97.0, 17.0
        ],
    );

    #[test]
    fn magnitude_percentage_ratio_plateau() {
        let fe = feat_extr!(MagnitudePercentageRatio::default());
        let x = [0.0; 10];
        let mut ts = TimeSeries::new_without_weight(&x, &x);
        assert_eq!(fe.eval(&mut ts), Err(EvaluatorError::FlatTimeSeries));
    }
}
