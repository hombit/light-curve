pub use crate::extractor::FeatureExtractor;
pub use crate::float_trait::Float;

pub use light_curve_common::{all_close, linspace};
pub use ndarray::Array1;
pub use rand::prelude::*;
pub use rand_distr::StandardNormal;

#[macro_export]
macro_rules! feature_test {
    ($name: ident, $fe: tt, $desired: expr, $y: expr $(,)?) => {
        feature_test!($name, $fe, $desired, $y, $y);
    };
    ($name: ident, $fe: tt, $desired: expr, $x: expr, $y: expr $(,)?) => {
        feature_test!($name, $fe, $desired, $x, $y, Array1::ones($x.len()));
    };
    ($name: ident, $fe: tt, $desired: expr, $x: expr, $y: expr, $w: expr $(,)?) => {
        feature_test!($name, $fe, $desired, $x, $y, $w, 1e-6);
    };
    ($name: ident, $fe: tt, $desired: expr, $x: expr, $y: expr, $w: expr, $tol: expr $(,)?) => {
        #[test]
        fn $name() {
            let fe = FeatureExtractor::new(vec!$fe);
            let desired = $desired;
            let x = $x;
            let y = $y;
            let w = $w;
            let mut ts = TimeSeries::new(&x, &y, &w);
            let actual = fe.eval(&mut ts).unwrap();
            all_close(&desired[..], &actual[..], $tol);

            let names = fe.get_names();
            let descs = fe.get_descriptions();
            assert_eq!(fe.size_hint(), actual.len(), "size_hint() returns wrong size");
            assert_eq!(actual.len(), names.len(),
                "Length of values and names should be the same");
            assert_eq!(actual.len(), descs.len(),
                "Length of values and descriptions should be the same");
        }
    };
}

#[macro_export]
macro_rules! eval_info_test {
    ($name: ident, $eval: expr $(,)?) => {
        #[test]
        fn $name() {
            const N: usize = 128;

            let mut rng = StdRng::seed_from_u64(0);

            let t1 = randvec::<f64>(&mut rng, N);
            let t1_sorted = sorted(&t1);
            assert_ne!(t1, t1_sorted);
            let m1 = randvec::<f64>(&mut rng, N);
            let w1 = positive_randvec::<f64>(&mut rng, N);

            let eval: Box<dyn FeatureEvaluator<f64>> = Box::new($eval);
            let size_hint = eval.size_hint();
            assert_eq!(
                eval.get_names().len(),
                size_hint,
                "names vector has a wrong size"
            );
            assert_eq!(
                eval.get_descriptions().len(),
                size_hint,
                "description vector has a wrong size"
            );
            let check_size =
                |v: &Vec<f64>| assert_eq!(size_hint, v.len(), "size_hint() returns wrong value");

            let baseline = eval
                .eval(&mut TimeSeries::new(&t1_sorted, &m1, &w1))
                .unwrap();
            check_size(&baseline);

            let min_ts_length = eval.min_ts_length();
            for n in (0..10) {
                let mut ts = TimeSeries::new(&t1_sorted[..n], &m1[..n], &w1[..n]);
                let result = eval.eval(&mut ts);
                let _ = result.as_ref().map(check_size);
                assert_eq!(
                    n >= min_ts_length,
                    result.is_ok(),
                    "min_ts_length() returns wrong value, \
                    time-series length: {}, \
                    min_ts_length(): {}, \
                    eval(ts).is_ok(): {}",
                    n,
                    min_ts_length,
                    result.is_ok(),
                );
            }

            {
                let t2 = randvec::<f64>(&mut rng, N);
                let t2_sorted = sorted(&t2);
                assert_ne!(t1_sorted, t2_sorted);

                let mut ts = TimeSeries::new(&t2_sorted, &m1, &w1);

                let v = eval.eval(&mut ts).unwrap();
                check_size(&v);
                let neq_baseline = !simeq(&v, &baseline, 1e-12);
                assert_eq!(
                    neq_baseline,
                    eval.is_t_required(),
                    "is_t_required() returns wrong value, \
                    v != baseline: {} ({:?} <=> {:?}), \
                    is_t_required(): {}",
                    neq_baseline,
                    v,
                    baseline,
                    eval.is_t_required(),
                );
            }

            {
                let m2 = randvec::<f64>(&mut rng, N);
                assert_ne!(m1, m2);

                let mut ts = TimeSeries::new(&t1_sorted, &m2, &w1);

                let v = eval.eval(&mut ts).unwrap();
                check_size(&v);
                let neq_baseline = !simeq(&v, &baseline, 1e-12);
                assert_eq!(
                    neq_baseline,
                    eval.is_m_required(),
                    "is_m_required() returns wrong value, \
                    v != baseline: {} ({:?} <=> {:?}), \
                    is_m_required(): {}",
                    neq_baseline,
                    v,
                    baseline,
                    eval.is_m_required(),
                );
            }

            {
                let w2 = positive_randvec::<f64>(&mut rng, N);
                assert_ne!(w1, w2);

                let mut ts = TimeSeries::new(&t1_sorted, &m1, &w2);
                let v = eval.eval(&mut ts).unwrap();
                check_size(&v);
                let neq_baseline = !simeq(&v, &baseline, 1e-12);
                assert_eq!(
                    neq_baseline,
                    eval.is_w_required(),
                    "is_w_required() returns wrong value, \
                    v != baseline: {}, \
                    is_w_required(): {}",
                    neq_baseline,
                    eval.is_w_required(),
                );
            }

            {
                let m1_ordered = sorted_by(&m1, &t1);
                assert_ne!(m1_ordered, m1);
                let w1_ordered = sorted_by(&w1, &t1);
                assert_ne!(w1_ordered, w1);

                let mut ts = TimeSeries::new(&t1, &m1_ordered, &w1_ordered);
                let v = eval.eval(&mut ts).unwrap();
                check_size(&v);
                let neq_baseline = !simeq(&v, &baseline, 1e-12);
                assert_eq!(
                    neq_baseline,
                    eval.is_sorting_required(),
                    "is_sorting_required() returns wrong value, \
                    unsorted result: {:?}, \
                    sorted result: {:?}, \
                    is_sorting_required: {}",
                    v,
                    baseline,
                    eval.is_sorting_required()
                );
            }
        }
    };
}

pub fn simeq<T: Float>(a: &[T], b: &[T], eps: T) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .all(|(&x, &y)| (x - y).abs() < eps + T::max(x.abs(), y.abs()) * eps)
}

pub fn randvec<T>(rng: &mut StdRng, n: usize) -> Vec<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    (0..n)
        .map(|_| {
            let x: T = rng.sample(StandardNormal);
            x
        })
        .collect()
}

pub fn positive_randvec<T>(rng: &mut StdRng, n: usize) -> Vec<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    let mut v = randvec(rng, n);
    v.iter_mut().for_each(|x| *x = x.abs());
    v
}

pub fn sorted<T>(v: &[T]) -> Vec<T>
where
    T: Float,
{
    let mut v = v.to_vec();
    v[..].sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

pub fn sorted_by<T: Float>(to_sort: &[T], key: &[T]) -> Vec<T> {
    assert_eq!(to_sort.len(), key.len());
    let mut idx: Vec<_> = (0..to_sort.len()).collect();
    idx[..].sort_by(|&a, &b| key[a].partial_cmp(&key[b]).unwrap());
    idx.iter().map(|&i| to_sort[i]).collect()
}
