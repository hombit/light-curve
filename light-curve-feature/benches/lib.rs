#[macro_use]
extern crate criterion;

mod extractor;
use extractor::bench_extractor;

mod fft_crates;
use fft_crates::bench_fft;

mod fit;
use fit::bench_fit_straight_line;

mod periodogram;
use periodogram::bench_periodogram;

mod recurrent_sin_cos;
use recurrent_sin_cos::bench_recurrent_sin_cos;

mod statistics;
use statistics::bench_peak_indices;

criterion_group!(benches_extractor, bench_extractor<f64>);
criterion_group!(benches_fft, bench_fft<f32>, bench_fft<f64>);
criterion_group!(benches_fit, bench_fit_straight_line);
criterion_group!(benches_periodogram, bench_periodogram);
criterion_group!(benches_recurrent_sin_cos, bench_recurrent_sin_cos);
criterion_group!(benches_statistics, bench_peak_indices);
criterion_main!(
    benches_extractor,
    benches_fft,
    benches_fit,
    benches_periodogram,
    benches_recurrent_sin_cos,
    benches_statistics
);
