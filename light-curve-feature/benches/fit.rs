use criterion::{black_box, Criterion};
use light_curve_common::linspace;
use light_curve_feature::{fit_straight_line, TimeSeries};
use rand::prelude::*;

pub fn bench_fit_straight_line(c: &mut Criterion) {
    const N: usize = 1000;

    let x = linspace(0.0_f64, 1.0, N);
    let y: Vec<_> = x.iter().map(|&x| x + thread_rng().gen::<f64>()).collect();
    let w: Vec<_> = x
        .iter()
        .map(|_| thread_rng().gen_range(10.0, 100.0))
        .collect();
    let ts = TimeSeries::new(&x, &y, &w);

    c.bench_function("Straight line fit w/o noise", |b| {
        b.iter(|| fit_straight_line(black_box(&ts), false));
    });
    c.bench_function("Straight line fit w/ noise", |b| {
        b.iter(|| fit_straight_line(black_box(&ts), true));
    });
}
