use conv::ConvUtil;
use criterion::{black_box, Criterion};
use light_curve_feature::*;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::any::type_name;

pub fn bench_extractor<T>(c: &mut Criterion)
where
    T: Float + 'static,
    StandardNormal: Distribution<T>,
{
    const N: [usize; 2] = [100, 1000];

    let features: Vec<Box<dyn FeatureEvaluator<_>>> = vec![
        Box::new(Amplitude::default()),
        Box::new(AndersonDarlingNormal::default()),
        Box::new(BeyondNStd::default()),
        Box::new(Cusum::default()),
        Box::new(Eta::default()),
        Box::new(EtaE::default()),
        Box::new(ExcessVariance::default()),
        Box::new(InterPercentileRange::default()),
        Box::new(Kurtosis::default()),
        Box::new(LinearFit::default()),
        Box::new(LinearTrend::default()),
        Box::new(MagnitudePercentageRatio::default()),
        Box::new(MaximumSlope::default()),
        Box::new(Mean::default()),
        Box::new(MeanVariance::default()),
        Box::new(Median::default()),
        Box::new(MedianAbsoluteDeviation::default()),
        Box::new(MedianBufferRangePercentage::default()),
        Box::new(PercentAmplitude::default()),
        Box::new(PercentDifferenceMagnitudePercentile::default()),
        Box::new(ReducedChi2::default()),
        Box::new(Skew::default()),
        Box::new(StandardDeviation::default()),
        Box::new(StetsonK::default()),
        Box::new(WeightedMean::default()),
    ];

    let observation_count_vec: Vec<_> = (0..20)
        .map(|_| {
            let f: Box<dyn FeatureEvaluator<T>> =
                Box::new(antifeatures::ObservationCount::default());
            f
        })
        .collect();

    let beyond_n_std_vec: Vec<_> = (1..21)
        .map(|i| {
            let f: Box<dyn FeatureEvaluator<_>> =
                Box::new(BeyondNStd::new(i.value_as::<T>().unwrap() / T::ten()));
            f
        })
        .collect();

    let mut bins = Bins::default();
    bins.add_feature(Box::new(StetsonK::default()));

    let mut periodogram = Periodogram::default();
    periodogram.set_max_freq_factor(10.0);

    let names_fes: Vec<_> = features
        .iter()
        .map(|f| (f.get_names()[0], FeatureExtractor::new(vec![f.clone()])))
        .chain(std::iter::once((
            "all non-meta features",
            FeatureExtractor::new(features.clone()),
        )))
        .chain(std::iter::once((
            "multiple ObservationCount",
            FeatureExtractor::new(observation_count_vec),
        )))
        .chain(std::iter::once((
            "multiple BeyondNStd",
            FeatureExtractor::new(beyond_n_std_vec),
        )))
        .chain(std::iter::once((
            "Bins",
            FeatureExtractor::new(vec![Box::new(bins)]),
        )))
        .chain(std::iter::once((
            "Periodogram",
            FeatureExtractor::new(vec![Box::new(periodogram)]),
        )))
        .collect();

    for &n in N.iter() {
        let x = randspace(n);
        let y = randvec(n);
        let err = randvec(n);
        for (name, fe) in names_fes.iter() {
            c.bench_function(
                format!("FeatureExtractor {}: [{}; {}]", name, n, type_name::<T>()).as_str(),
                |b| {
                    b.iter(|| {
                        run(black_box(fe), black_box(&x), black_box(&y), black_box(&err)).unwrap();
                    });
                },
            );
        }
    }
}

fn run<T: Float>(
    fe: &FeatureExtractor<T>,
    x: &[T],
    y: &[T],
    err: &[T],
) -> Result<Vec<T>, EvaluatorError> {
    let w: Vec<_> = err.iter().map(|&e| e.powi(-2)).collect();
    let mut ts = TimeSeries::new(&x, &y, Some(&w));
    fe.eval(&mut ts)
}

fn randvec<T>(n: usize) -> Vec<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    (0..n)
        .map(|_| {
            let x: T = thread_rng().sample(StandardNormal);
            x
        })
        .collect()
}

fn randspace<T>(n: usize) -> Vec<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    let mut x = randvec::<T>(n);
    x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    x
}
