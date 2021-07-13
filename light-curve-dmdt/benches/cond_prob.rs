use criterion::{black_box, Criterion};
use light_curve_dmdt::{DmDt, Eps1Over1e3Erf};
use ndarray::Array1;

pub fn bench_cond_prob(c: &mut Criterion) {
    let dmdt = DmDt::from_lgdt_dm_limits(0.0_f32, 2.0_f32, 32, 1.25_f32, 32);

    let t = Array1::linspace(0.0, 100.0, 101);
    let m = t.mapv(f32::sin);
    // err is ~0.03
    let err2 = Array1::from_elem(101, 0.001);

    c.bench_function(
        "conditional probability = convert_lc_to_gausses() / dt_points()",
        |b| {
            b.iter(|| {
                let mut map = dmdt.gausses::<Eps1Over1e3Erf>(
                    t.as_slice().unwrap(),
                    m.as_slice().unwrap(),
                    err2.as_slice().unwrap(),
                );
                let dt_points = dmdt.dt_points(t.as_slice().unwrap());
                let dt_points_no_zeros = dt_points.mapv(|x| if x == 0 { 1.0 } else { x as f32 });
                map /= &dt_points_no_zeros.into_shape((map.nrows(), 1)).unwrap();
                black_box(map);
            })
        },
    );
    c.bench_function("conditional probability = cond_prob()", |b| {
        b.iter(|| {
            black_box(dmdt.cond_prob::<Eps1Over1e3Erf>(
                t.as_slice().unwrap(),
                m.as_slice().unwrap(),
                err2.as_slice().unwrap(),
            ));
        })
    });
}
