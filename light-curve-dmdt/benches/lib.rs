#[macro_use]
extern crate criterion;

mod cond_prob;
use cond_prob::bench_cond_prob;

mod erf;
use erf::{bench_erf, bench_erfinv};

mod gausses;
use gausses::bench_gausses;

mod grid;
use grid::bench_log_linear_grids;

criterion_group!(benches_cond_prob, bench_cond_prob);
criterion_group!(
    benches_erf,
    bench_erf<f32>,
    bench_erf<f64>,
    bench_erfinv<f32>,
    bench_erfinv<f64>
);
criterion_group!(benches_gausses, bench_gausses<f32>, bench_gausses<f64>);
criterion_group!(
    benches_grid,
    bench_log_linear_grids<f32>,
    bench_log_linear_grids<f64>
);

criterion_main!(
    benches_cond_prob,
    benches_erf,
    benches_gausses,
    benches_grid
);
