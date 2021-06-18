use ndarray::{CowArray, Ix1};

pub type CowArray1<'a, T> = CowArray<'a, T, Ix1>;
