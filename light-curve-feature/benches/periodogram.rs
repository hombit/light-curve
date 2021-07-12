use criterion::{black_box, Criterion};
use light_curve_common::linspace;
use light_curve_feature::periodogram::*;
use light_curve_feature::time_series::TimeSeries;

type BoxedPeriodogram = Box<dyn PeriodogramPower<f32>>;

pub fn bench_periodogram(c: &mut Criterion) {
    let ns_power_resolution: [(Vec<usize>, BoxedPeriodogram, f32); 2] = [
        (vec![10, 100, 1000], Box::new(PeriodogramPowerDirect), 5.0),
        (
            vec![10, 100, 1000, 10000, 1000000],
            Box::new(PeriodogramPowerFft::new()),
            10.0,
        ),
    ];
    const PERIOD: f32 = 0.22;
    let nyquist: Box<dyn NyquistFreq<f32>> = Box::new(AverageNyquistFreq);

    for (ns, power, resolution) in ns_power_resolution.iter() {
        for &n in ns {
            let x = linspace(0.0_f32, 1.0, n);
            let y: Vec<_> = x
                .iter()
                .map(|&x| 3.0 * f32::sin(2.0 * std::f32::consts::PI / PERIOD * x + 0.5) + 4.0)
                .collect();
            c.bench_function(
                format!("Periodogram: {} length, {:?}", n, power).as_str(),
                |b| {
                    b.iter(|| {
                        let mut ts = TimeSeries::new_without_weight(&x, &y);
                        let periodogram =
                            Periodogram::from_t(power.clone(), &x, *resolution, 1.0, &nyquist);
                        periodogram.power(black_box(&mut ts));
                    })
                },
            );
        }
    }
}
