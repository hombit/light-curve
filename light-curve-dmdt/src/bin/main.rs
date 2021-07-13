use clap::{value_t, App, Arg, ArgMatches};
use enumflags2::{bitflags, BitFlags};
use light_curve_dmdt::{to_png, DmDt, Eps1Over1e3Erf, ExactErf};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::str::FromStr;
use thiserror::Error;

#[derive(Error, std::fmt::Debug)]
#[error("{0}")]
enum MainError {
    IoError(#[from] std::io::Error),
    ParseFloatError(#[from] std::num::ParseFloatError),
    PngEncodingError(#[from] png::EncodingError),
    NotEnoughColumns(&'static str),
}

fn main() -> Result<(), MainError> {
    let config = Config::from_arg_matches(&arg_matches());

    let (t, m, err2) = read_input(&config.input, config.smearing)?;

    let dmdt = DmDt::from_lgdt_dm_limits(
        config.min_lgdt,
        config.max_lgdt,
        config.n_dt,
        config.max_abs_dm,
        config.n_dm,
    );

    let map_float_or_u8 = if config.smearing {
        let map_float = match config.approx_smearing {
            true => dmdt.gausses::<Eps1Over1e3Erf>(&t, &m, &err2.unwrap()),
            false => dmdt.gausses::<ExactErf>(&t, &m, &err2.unwrap()),
        };
        Array2FloatOrU8::Float(map_float)
    } else {
        let map_usize = dmdt.points(&t, &m);
        if config.norm.is_empty() {
            Array2FloatOrU8::U8(map_usize.mapv(|x| x.clamp(0, 255) as u8))
        } else {
            Array2FloatOrU8::Float(map_usize.mapv(|x| x as f32))
        }
    };
    let map_u8 = match map_float_or_u8 {
        Array2FloatOrU8::U8(map_u8) => map_u8,
        Array2FloatOrU8::Float(mut map_float) => {
            if config.norm.contains(DmDtNorm::LgDt) {
                let lgdt = dmdt.dt_points(&t);
                let lgdt_no_zeros = lgdt.mapv(|x| if x == 0 { 1.0 } else { x as f32 });
                map_float /= &lgdt_no_zeros.into_shape((map_float.nrows(), 1)).unwrap();
            }
            if config.norm.contains(DmDtNorm::Max) {
                let max = *map_float
                    .iter()
                    .max_by(|&x, &y| x.partial_cmp(y).unwrap())
                    .unwrap();
                map_float /= max;
            }
            if !config.norm.is_empty() {
                map_float *= 255.0;
            }
            map_float.mapv(|x| x.clamp(0.0, 255.0) as u8)
        }
    };

    let stdout = std::io::stdout();
    let writer: Box<dyn Write> = match &config.output {
        Some(path) => Box::new(BufWriter::new(File::create(path)?)),
        None => Box::new(stdout.lock()),
    };
    to_png(writer, &map_u8)?;

    Ok(())
}

enum Array2FloatOrU8 {
    Float(ndarray::Array2<f32>),
    U8(ndarray::Array2<u8>),
}

#[bitflags]
#[repr(u8)]
#[derive(Clone, Copy)]
pub enum DmDtNorm {
    LgDt,
    Max,
}

type Tme2 = (Vec<f32>, Vec<f32>, Option<Vec<f32>>);

fn read_input(input: &Option<PathBuf>, errors: bool) -> Result<Tme2, MainError> {
    let mut t = vec![];
    let mut m = vec![];
    let mut err2 = if errors { Some(vec![]) } else { None };
    let stdin = std::io::stdin();
    let buffer: Box<dyn BufRead> = match input {
        Some(path) => Box::new(BufReader::new(File::open(path)?)),
        None => Box::new(stdin.lock()),
    };
    for line in buffer.lines() {
        let line = line?;
        if line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        t.push(f32::from_str(
            it.next()
                .ok_or(MainError::NotEnoughColumns("Empty string"))?,
        )?);
        m.push(f32::from_str(it.next().ok_or(
            MainError::NotEnoughColumns("Only one value in line, at least two required"),
        )?)?);
        if errors {
            err2.as_mut().unwrap().push(f32::powi(
                f32::from_str(it.next().ok_or(MainError::NotEnoughColumns(
                    "Only two values in a line, at least three required",
                ))?)?,
                2,
            ));
        }
    }
    Ok((t, m, err2))
}

fn arg_matches() -> ArgMatches<'static> {
    App::new("Plot dm-dt map from light curve")
        .arg(
            Arg::with_name("input")
                .short("i")
                .long("input")
                .takes_value(true)
                .default_value("-")
                .long_help(
                    "Path of the input file, should be built of space-separated columns of \
                    time, magnitude and magnitude error (required for --smare only). If '-' is \
                    given (the default), then the input is taken from the stdin",
                ),
        )
        .arg(
            Arg::with_name("output")
                .short("o")
                .long("output")
                .takes_value(true)
                .default_value("-")
                .long_help(
                    "Path of the output PNG file. If '-' is given (the default), then outputs \
                    to the stdout",
                ),
        )
        .arg(
            Arg::with_name("smear")
                .short("s")
                .long("smear")
                .takes_value(false)
                .help("dm smearing")
                .long_help(
                    "Produce dm-``smeared'' output using observation errors, which must be the \
                    third column of the input. Instead of just adding some value to the lg(dt)-dm \
                    cell, the whole lg(dt) = const row is filled by normally distributed \
                    dm-probabilities",
                ),
        )
        .arg(
            Arg::with_name("min lgdt")
                .long("min-lgdt")
                .takes_value(true)
                .required(true)
                .help("left lg(dt) border")
                .long_help(
                    "Left border of the lg(dt) grid, note that decimal logarithm is required, \
                    i.e. -1.0 input means 0.1 time units",
                ),
        )
        .arg(
            Arg::with_name("max lgdt")
                .long("max-lgdt")
                .takes_value(true)
                .required(true)
                .help("right lg(dt) border")
                .long_help(
                    "Right border of the lg(dt) grid, note that decimal logarithm is required, \
                    i.e. 2.0 input means 100.0 time units",
                ),
        )
        .arg(
            Arg::with_name("max abs dm")
                .long("max-abs-dm")
                .takes_value(true)
                .required(true)
                .help("absolute value of dm border")
                .long_help(
                    "Maximum dm value, the considered dm interval would be \
                    [-max-abs-dm, +max-abs-dm)",
                ),
        )
        .arg(
            Arg::with_name("N lgdt")
                .short("w")
                .long("width")
                .takes_value(true)
                .default_value("128")
                .help("number of lg(dt) cells, width of the output image"),
        )
        .arg(
            Arg::with_name("N dm")
                .short("h")
                .long("height")
                .takes_value(true)
                .default_value("128")
                .help("number of dm cells, height of the output image"),
        )
        .arg(
            Arg::with_name("approx smearing")
                .long("approx-smearing")
                .takes_value(false)
                .help("speed up smearing using approximate error function"),
        )
        .arg(
            Arg::with_name("normalisation")
                .short("n")
                .long("norm")
                .takes_value(true)
                .multiple(true)
                .possible_values(&["lgdt", "max"])
                .help("normalisation, any combination of: max, lgdt")
                .long_help(
                    "Normalisation to do after dmdt map building. The order \
                    of operations is:\
                    1) build dmdt map, each dm-lgdt pair brings a unity value \
                    to dmdt space;\
                    2) if --norm=lgdt, then divide each cell value by the \
                    total number of the corresponding lgdt pairs, i.e. divide \
                    each cell of some column by the integral value in the \
                    column (including values out of the interval of \
                    [-max_abs_dm; max_abs_dm)); \
                    3) if --norm=max, then divide each cell by the overall \
                    maximum value; \
                    4) if any of --norm=lgdt or --norm=max is specified, then \
                    all values should be in [0; 1] interval, so they are \
                    multiplied by 255 and casted to uint8 to make it possible \
                    to save dmdt map as a PNG file.",
                ),
        )
        .get_matches()
}

struct Config {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    min_lgdt: f32,
    max_lgdt: f32,
    max_abs_dm: f32,
    n_dt: usize,
    n_dm: usize,
    smearing: bool,
    approx_smearing: bool,
    norm: BitFlags<DmDtNorm>,
}

impl Config {
    fn from_arg_matches(matches: &ArgMatches) -> Self {
        Self {
            input: match matches.value_of("input").unwrap() {
                "-" => None,
                x => Some(x.into()),
            },
            output: match matches.value_of("output").unwrap() {
                "-" => None,
                x => Some(x.into()),
            },
            min_lgdt: value_t!(matches, "min lgdt", f32).unwrap(),
            max_lgdt: value_t!(matches, "max lgdt", f32).unwrap(),
            max_abs_dm: value_t!(matches, "max abs dm", f32).unwrap(),
            n_dt: value_t!(matches, "N lgdt", usize).unwrap(),
            n_dm: value_t!(matches, "N dm", usize).unwrap(),
            smearing: matches.is_present("smear"),
            approx_smearing: matches.is_present("approx smearing"),
            norm: match matches.values_of("normalisation") {
                Some(values) => values
                    .map(|s| match s {
                        "lgdt" => DmDtNorm::LgDt,
                        "max" => DmDtNorm::Max,
                        _ => panic!("the normalisation '{}' is not supported", s),
                    })
                    .collect(),
                None => BitFlags::empty(),
            },
        }
    }
}
