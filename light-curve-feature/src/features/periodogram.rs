use crate::evaluator::*;
use crate::extractor::FeatureExtractor;
use crate::periodogram;
use crate::periodogram::{AverageNyquistFreq, NyquistFreq, PeriodogramPower, PeriodogramPowerFft};
use crate::statistics::Statistics;
use std::iter;

fn number_ending(i: usize) -> &'static str {
    match (i % 10, i % 100) {
        (1, 11) => "th",
        (1, _) => "st",
        (2, 12) => "th",
        (2, _) => "nd",
        (3, 13) => "th",
        (3, _) => "rd",
        (_, _) => "th",
    }
}

/// Peak evaluator for `Periodogram`
#[derive(Clone, Debug)]
struct PeriodogramPeaks {
    info: EvaluatorInfo,
    peaks: usize,
    names: Vec<String>,
    descriptions: Vec<String>,
}

impl PeriodogramPeaks {
    fn new(peaks: usize) -> Self {
        assert!(peaks > 0, "Number of peaks should be at least one");
        Self {
            info: EvaluatorInfo {
                size: 2 * peaks,
                min_ts_length: 1,
                t_required: true,
                m_required: true,
                w_required: false,
                sorting_required: true,
            },
            peaks,
            names: (0..peaks)
                .flat_map(|i| vec![format!("period_{}", i), format!("period_s_to_n_{}", i)])
                .collect(),
            descriptions: (0..peaks)
                .flat_map(|i| {
                    vec![
                        format!(
                            "period of the {}{} highest peak of periodogram",
                            i + 1,
                            number_ending(i + 1),
                        ),
                        format!(
                            "Spectral density to spectral density standard deviation ratio of \
                            the {}{} highest peak of periodogram",
                            i + 1,
                            number_ending(i + 1)
                        ),
                    ]
                })
                .collect(),
        }
    }

    #[inline]
    fn default_peaks() -> usize {
        1
    }
}

impl Default for PeriodogramPeaks {
    fn default() -> Self {
        Self::new(Self::default_peaks())
    }
}

impl<T> FeatureEvaluator<T> for PeriodogramPeaks
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(ts
            .m
            .sample
            .as_slice()
            .unwrap()
            .peak_indices_reverse_sorted()
            .iter()
            .flat_map(|&i| {
                iter::once(Periodogram::period(ts.t.sample[i]))
                    .chain(iter::once(ts.m.signal_to_noise(ts.m.sample[i])))
            })
            .chain(iter::repeat(T::zero()))
            .take(2 * self.peaks)
            .collect())
    }

    fn get_info(&self) -> &EvaluatorInfo {
        &self.info
    }

    fn get_names(&self) -> Vec<&str> {
        self.names.iter().map(|name| name.as_str()).collect()
    }

    fn get_descriptions(&self) -> Vec<&str> {
        self.descriptions.iter().map(|desc| desc.as_str()).collect()
    }
}

// See http://doi.org/10.1088/0004-637X/733/1/10
/// A number of features based on Lomb–Scargle periodogram
///
/// Periodogram $P(\omega)$ is an estimate of spectral density of unevenly time series.
/// `Periodogram::new`'s `peaks` argument corresponds to a number of the most significant spectral
/// density peaks to return. For each peak its period and "signal to noise" ratio is returned.
///
/// $$
/// \mathrm{signal~to~noise~of~peak} \equiv \frac{P(\omega_\mathrm{peak}) - \langle P(\omega) \rangle}{\sigma\_{P(\omega)}}.
/// $$
///
/// `Periodogram` can accept another `dyn FeatureEvaluator` for feature extraction from periodogram
/// as it was time series without observation errors. You can even pass one `Periodogram` to another
/// one if you are crazy enough
///
/// - Depends on: **time**, **magnitude**
/// - Minimum number of observations: **2** (or as required by sub-features)
/// - Number of features: **$2 \times \mathrm{peaks}~+...$**
#[derive(Clone, Debug)]
pub struct Periodogram<T: Float> {
    info: EvaluatorInfo,
    resolution: f32,
    max_freq_factor: f32,
    nyquist: Box<dyn NyquistFreq<T>>,
    feature_extractor: FeatureExtractor<T>,
    feature_names: Vec<String>,
    feature_descriptions: Vec<String>,
    periodogram_algorithm: Box<dyn PeriodogramPower<T>>,
}

impl<T> Periodogram<T>
where
    T: Float,
{
    #[inline]
    pub fn default_peaks() -> usize {
        PeriodogramPeaks::default_peaks()
    }

    #[inline]
    pub fn default_resolution() -> f32 {
        10.0
    }

    #[inline]
    pub fn default_max_freq_factor() -> f32 {
        1.0
    }

    /// New [Periodogram] that finds given number of peaks
    pub fn new(peaks: usize) -> Self {
        let peaks = PeriodogramPeaks::new(peaks);
        let peak_names = peaks.names.clone();
        let peak_descriptions = peaks.descriptions.clone();
        let peaks_evaluator = &peaks as &dyn FeatureEvaluator<T>;
        let peaks_size_hint = peaks_evaluator.size_hint();
        let peaks_min_ts_length = peaks_evaluator.min_ts_length();
        Self {
            info: EvaluatorInfo {
                size: peaks_size_hint,
                min_ts_length: usize::max(peaks_min_ts_length, 2),
                t_required: true,
                m_required: true,
                w_required: false,
                sorting_required: true,
            },
            resolution: Self::default_resolution(),
            max_freq_factor: Self::default_max_freq_factor(),
            nyquist: Box::new(AverageNyquistFreq),
            feature_extractor: feat_extr!(peaks),
            feature_names: peak_names,
            feature_descriptions: peak_descriptions,
            periodogram_algorithm: Box::new(PeriodogramPowerFft::new()),
        }
    }

    /// Set frequency resolution
    ///
    /// The larger frequency resolution allows to find peak period with better precision
    pub fn set_freq_resolution(&mut self, resolution: f32) -> &mut Self {
        self.resolution = resolution;
        self
    }

    /// Multiply maximum (Nyquist) frequency
    ///
    /// Maximum frequency is Nyquist frequncy multiplied by this factor. The larger factor allows
    /// to find larger frequency and makes [PeriodogramPowerFft] more precise. However large
    /// frequencies can show false peaks
    pub fn set_max_freq_factor(&mut self, max_freq_factor: f32) -> &mut Self {
        self.max_freq_factor = max_freq_factor;
        self
    }

    /// Define Nyquist frequency
    pub fn set_nyquist(&mut self, nyquist: Box<dyn NyquistFreq<T>>) -> &mut Self {
        self.nyquist = nyquist;
        self
    }

    /// Extend a feature to extract from periodogram
    pub fn add_feature(&mut self, feature: Box<dyn FeatureEvaluator<T>>) -> &mut Self {
        self.info.size += feature.size_hint();
        self.feature_names.extend(
            feature
                .get_names()
                .iter()
                .map(|name| "periodogram_".to_owned() + name),
        );
        self.feature_descriptions.extend(
            feature
                .get_descriptions()
                .into_iter()
                .map(|desc| format!("{} of periodogram", desc)),
        );
        self.feature_extractor.add_feature(feature);
        self
    }

    pub fn set_periodogram_algorithm(
        &mut self,
        periodogram_power: Box<dyn PeriodogramPower<T>>,
    ) -> &mut Self {
        self.periodogram_algorithm = periodogram_power;
        self
    }

    fn periodogram(&self, ts: &mut TimeSeries<T>) -> periodogram::Periodogram<T> {
        periodogram::Periodogram::from_t(
            self.periodogram_algorithm.clone(),
            ts.t.sample.as_slice().unwrap(),
            self.resolution,
            self.max_freq_factor,
            &self.nyquist,
        )
    }

    pub fn power(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        self.periodogram(ts).power(ts)
    }

    pub fn freq_power(&self, ts: &mut TimeSeries<T>) -> (Vec<T>, Vec<T>) {
        let p = self.periodogram(ts);
        let power = p.power(ts);
        let freq: Vec<_> = (0..power.len()).map(|i| p.freq(i)).collect();
        (freq, power)
    }

    fn transform_ts(&self, ts: &mut TimeSeries<T>) -> Result<TmwVectors<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let (freq, power) = self.freq_power(ts);
        Ok(TmwVectors {
            t: freq,
            m: power,
            w: None,
        })
    }

    fn period(omega: T) -> T {
        T::two() * T::PI() / omega
    }
}

impl<T> Default for Periodogram<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(Self::default_peaks())
    }
}

impl<T> FeatureEvaluator<T> for Periodogram<T>
where
    T: Float,
{
    transformer_eval!();
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::features::amplitude::Amplitude;
    use crate::periodogram::{PeriodogramPowerDirect, QuantileNyquistFreq};
    use crate::tests::*;

    eval_info_test!(periodogram_info_1, {
        let mut periodogram = Periodogram::default();
        periodogram.set_periodogram_algorithm(Box::new(PeriodogramPowerDirect {}));
        periodogram
    });

    eval_info_test!(periodogram_info_2, {
        let mut periodogram = Periodogram::new(5);
        periodogram.set_periodogram_algorithm(Box::new(PeriodogramPowerDirect {}));
        periodogram
    });

    eval_info_test!(periodogram_info_3, {
        let mut periodogram = Periodogram::default();
        periodogram.add_feature(Box::new(Amplitude::default()));
        periodogram.set_periodogram_algorithm(Box::new(PeriodogramPowerDirect {}));
        periodogram
    });

    #[test]
    fn periodogram_plateau() {
        let fe = FeatureExtractor::new(vec![Box::new(Periodogram::default())]);
        let x = linspace(0.0_f32, 1.0, 100);
        let y = [0.0_f32; 100];
        let mut ts = TimeSeries::new(&x[..], &y[..], None);
        let desired = vec![0.0, 0.0];
        let actual = fe.eval(&mut ts).unwrap();
        assert_eq!(desired, actual);
    }

    #[test]
    fn periodogram_evenly_sinus() {
        let fe = FeatureExtractor::new(vec![Box::new(Periodogram::default())]);
        let mut rng = StdRng::seed_from_u64(0);
        let period = 0.17;
        let x = linspace(0.0_f32, 1.0, 101);
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period * x + 0.5)
                    + 4.0
                    + 0.01 * rng.gen::<f32>() // noise stabilizes solution
            })
            .collect();
        let mut ts = TimeSeries::new(&x[..], &y[..], None);
        let desired = [period];
        let actual = [fe.eval(&mut ts).unwrap()[0]]; // Test period only
        all_close(&desired[..], &actual[..], 5e-3);
    }

    #[test]
    fn periodogram_unevenly_sinus() {
        let fe = FeatureExtractor::new(vec![Box::new(Periodogram::default())]);
        let period = 0.17;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..100).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| 3.0 * f32::sin(2.0 * std::f32::consts::PI / period * x + 0.5) + 4.0)
            .collect();
        let mut ts = TimeSeries::new(&x[..], &y[..], None);
        let desired = [period];
        let actual = [fe.eval(&mut ts).unwrap()[0]]; // Test period only
        all_close(&desired[..], &actual[..], 5e-3);
    }

    #[test]
    fn periodogram_one_peak_vs_two_peaks() {
        let fe = FeatureExtractor::new(vec![
            Box::new(Periodogram::new(1)),
            Box::new(Periodogram::new(2)),
        ]);
        let period = 0.17;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..100).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| 3.0 * f32::sin(2.0 * std::f32::consts::PI / period * x + 0.5) + 4.0)
            .collect();
        let mut ts = TimeSeries::new(&x[..], &y[..], None);
        let features = fe.eval(&mut ts).unwrap();
        all_close(
            &[features[0], features[1]],
            &[features[2], features[3]],
            1e-6,
        );
    }

    #[test]
    fn periodogram_unevenly_sinus_cosine() {
        let fe = FeatureExtractor::new(vec![Box::new(Periodogram::new(2))]);
        let period1 = 0.0753;
        let period2 = 0.45;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..1000).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period1 * x + 0.5)
                    + -5.0 * f32::cos(2.0 * std::f32::consts::PI / period2 * x + 0.5)
                    + 4.0
            })
            .collect();
        let mut ts = TimeSeries::new(&x[..], &y[..], None);
        let desired = [period2, period1];
        let features = fe.eval(&mut ts).unwrap();
        let actual = [features[0], features[2]]; // Test period only
        all_close(&desired[..], &actual[..], 1e-2);
        assert!(features[1] > features[3]);
    }

    #[test]
    fn periodogram_unevenly_sinus_cosine_noised() {
        let fe = FeatureExtractor::new(vec![Box::new(Periodogram::new(2))]);
        let period1 = 0.0753;
        let period2 = 0.46;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..1000).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period1 * x + 0.5)
                    + -5.0 * f32::cos(2.0 * std::f32::consts::PI / period2 * x + 0.5)
                    + 10.0 * rng.gen::<f32>()
                    + 4.0
            })
            .collect();
        let mut ts = TimeSeries::new(&x[..], &y[..], None);
        let desired = [period2, period1];
        let features = fe.eval(&mut ts).unwrap();
        let actual = [features[0], features[2]]; // Test period only
        all_close(&desired[..], &actual[..], 1e-2);
        assert!(features[1] > features[3]);
    }

    #[test]
    fn periodogram_different_time_scales() {
        let mut periodogram = Periodogram::new(2);
        periodogram
            .set_nyquist(Box::new(QuantileNyquistFreq { quantile: 0.05 }))
            .set_freq_resolution(10.0)
            .set_max_freq_factor(1.0)
            .set_periodogram_algorithm(Box::new(PeriodogramPowerFft::new()));
        let fe = FeatureExtractor::new(vec![Box::new(periodogram)]);
        let period1 = 0.01;
        let period2 = 1.0;
        let n = 100;
        let mut x = linspace(0.0, 0.1, n);
        x.append(&mut linspace(1.0, 10.0, n));
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period1 * x + 0.5)
                    + -5.0 * f32::cos(2.0 * std::f32::consts::PI / period2 * x + 0.5)
                    + 4.0
            })
            .collect();
        let mut ts = TimeSeries::new(&x, &y, None);
        let features = fe.eval(&mut ts).unwrap();
        assert!(f32::abs(features[0] - period2) / period2 < 1.0 / n as f32);
        assert!(f32::abs(features[2] - period1) / period1 < 1.0 / n as f32);
        assert!(features[1] > features[3]);
    }
}
