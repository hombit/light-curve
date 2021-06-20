/// Helper for static EvaluatorInfo creation
#[macro_export]
macro_rules! lazy_info {
    (
        $name: ident,
        size: $size: expr,
        min_ts_length: $len: expr,
        t_required: $t: expr,
        m_required: $m: expr,
        w_required: $w: expr,
        sorting_required: $sort: expr,
    ) => {
        lazy_static! {
            static ref $name: EvaluatorInfo = EvaluatorInfo {
                size: $size,
                min_ts_length: $len,
                t_required: $t,
                m_required: $m,
                w_required: $w,
                sorting_required: $sort,
            };
        }
    };
}

/// Helper for FeatureEvaluator implementations using time-series transformation.
/// You must implement:
/// - method `transform_ts(ts: &mut TimeSeries<T>) -> TMWVectors<T>`
/// - attribute `info: EvaluatorInfo`
/// - attribute `feature_names: Vec<String>`
/// - attribute `feature_descriptions: Vec<String>`
#[macro_export]
macro_rules! transformer_eval {
    () => {
        fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
            let tmw = self.transform_ts(ts)?;
            let mut new_ts = TimeSeries::new(&tmw.t, &tmw.m, tmw.w.as_ref().map(|w| &w[..]));
            self.feature_extractor.eval(&mut new_ts)
        }

        fn eval_or_fill(&self, ts: &mut TimeSeries<T>, fill_value: T) -> Vec<T> {
            let tmw = match self.transform_ts(ts) {
                Ok(x) => x,
                Err(_) => return vec![fill_value; self.size_hint()],
            };
            let mut new_ts = TimeSeries::new(&tmw.t, &tmw.m, tmw.w.as_ref().map(|w| &w[..]));
            self.feature_extractor.eval_or_fill(&mut new_ts, fill_value)
        }

        fn get_info(&self) -> &EvaluatorInfo {
            &self.info
        }

        fn get_names(&self) -> Vec<&str> {
            self.feature_names
                .iter()
                .map(|name| name.as_str())
                .collect()
        }

        fn get_descriptions(&self) -> Vec<&str> {
            self.feature_descriptions
                .iter()
                .map(|desc| desc.as_str())
                .collect()
        }
    };
}
