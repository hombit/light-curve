use conv::*;
use criterion::{black_box, Criterion};
use light_curve_dmdt::{DmDt, Eps1Over1e3Erf, ErfFloat, ExactErf};
use ndarray::Array1;

pub fn bench_gausses<T>(c: &mut Criterion)
where
    T: ErfFloat + ValueFrom<f32>,
{
    let dmdt = DmDt::from_lgdt_dm_limits(
        T::zero(),
        2.0_f32.value_as::<T>().unwrap(),
        32,
        1.25_f32.value_as::<T>().unwrap(),
        32,
    );

    let t = Array1::linspace(T::zero(), 100.0_f32.value_as::<T>().unwrap(), 101);
    let m = t.mapv(T::sin);
    // err is ~0.03
    let err2 = Array1::from_elem(101, 0.001_f32.value_as::<T>().unwrap());

    let gausses_methods: [(_, Box<dyn Fn(_, _, _) -> _>); 2] = [
        (
            "ExactErf",
            Box::new(|t, m, err2| dmdt.gausses::<ExactErf>(t, m, err2)),
        ),
        (
            "Eps1Over1e3Erf",
            Box::new(|t, m, err2| dmdt.gausses::<Eps1Over1e3Erf>(t, m, err2)),
        ),
    ];

    for (erf, method) in gausses_methods.iter() {
        c.bench_function(
            format!(
                "DmDt<{}>::convert_lc_to_gausses, erf type is {}",
                std::any::type_name::<T>(),
                erf,
            )
            .as_str(),
            |b| {
                b.iter(|| {
                    black_box(method(
                        t.as_slice_memory_order().unwrap(),
                        m.as_slice_memory_order().unwrap(),
                        err2.as_slice_memory_order().unwrap(),
                    ));
                })
            },
        );
    }
}
