use conv::*;
use criterion::{black_box, Criterion};
use light_curve_dmdt::{ArrayGrid, DmDt, Float};
use ndarray::Array1;

pub fn bench_log_linear_grids<T>(c: &mut Criterion)
where
    T: Float + ValueFrom<f32>,
{
    let dmdt_lg_linear = DmDt::from_lgdt_dm_limits(
        T::zero(),
        2.0_f32.value_as::<T>().unwrap(),
        32,
        1.25_f32.value_as::<T>().unwrap(),
        32,
    );
    let dmdt_arrays: DmDt<T> = DmDt {
        dt_grid: Box::new(
            ArrayGrid::new(Array1::logspace(
                10.0_f32.value_as::<T>().unwrap(),
                0.0_f32.value_as::<T>().unwrap(),
                2.0_f32.value_as::<T>().unwrap(),
                33,
            ))
            .unwrap(),
        ),
        dm_grid: Box::new(
            ArrayGrid::new(Array1::linspace(
                -1.25_f32.value_as::<T>().unwrap(),
                1.25_f32.value_as::<T>().unwrap(),
                33,
            ))
            .unwrap(),
        ),
    };

    let t = Array1::linspace(T::zero(), 100.0_f32.value_as::<T>().unwrap(), 101);
    let m = t.mapv(T::sin);

    c.bench_function(
        format!(
            "DmDt<{}>::points, log and linear grids",
            std::any::type_name::<T>()
        )
        .as_str(),
        |b| {
            b.iter(|| {
                black_box(dmdt_lg_linear.points(t.as_slice().unwrap(), m.as_slice().unwrap()));
            })
        },
    );

    c.bench_function(
        format!("DmDt<{}>::points, array grids", std::any::type_name::<T>()).as_str(),
        |b| {
            b.iter(|| {
                black_box(dmdt_arrays.points(t.as_slice().unwrap(), m.as_slice().unwrap()));
            })
        },
    );
}
