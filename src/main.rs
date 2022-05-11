use prover::{
    crypto::{hashers::Rp64_256, ElementHasher},
    iterators::*,
    math::{
        fields::{f64::BaseElement as Felt, CubeExtension, QuadExtension},
        log2, FieldElement,
    },
    Air, AirContext, Assertion, EvaluationFrame, FieldExtension, HashFunction, Matrix,
    ProofOptions, StarkDomain, TraceInfo, TransitionConstraintDegree,
};
use rand_utils::rand_vector;
use std::time::Instant;
use structopt::StructOpt;

pub fn main() {
    // read command-line args
    let options = BenchOptions::from_args();

    match options.extension_degree {
        1 => run_benchmarks::<Felt, Rp64_256>(options),
        2 => run_benchmarks::<QuadExtension<Felt>, Rp64_256>(options),
        3 => run_benchmarks::<CubeExtension<Felt>, Rp64_256>(options),
        _ => panic!("invalid field extension option"),
    }
}

// BENCHMARK FUNCTIONS
// ================================================================================================

fn run_benchmarks<E, H>(options: BenchOptions)
where
    E: FieldElement<BaseField = Felt>,
    H: ElementHasher<BaseField = E::BaseField>,
{
    let now = Instant::now();
    let domain = build_domain(options.num_cols, options.log_n_rows, options.blowup);
    let trace = build_rand_matrix::<E>(options.num_cols, options.log_n_rows);
    //let trace = build_fib_matrix::<E>(options.num_cols, options.log_n_rows);
    println!(
        "prepared benchmark inputs in {:.2} sec",
        now.elapsed().as_millis() as f64 / 1000_f64
    );

    // perform interpolation
    let start = Instant::now();
    let polys = trace.interpolate_columns();
    let interpolate_result = start.elapsed().as_millis() as f64 / 1000_f64;

    // perform evaluation
    let extended_trace = polys.evaluate_columns_over(&domain);
    let lde_result = start.elapsed().as_millis() as f64 / 1000_f64;

    // build Merkle tree
    let mtree_start = Instant::now();
    let _tree = extended_trace.commit_to_rows::<H>();
    let mtree_result = mtree_start.elapsed().as_millis() as f64 / 1000_f64;
    let overall_result = start.elapsed().as_millis() as f64 / 1000_f64;

    // print out results
    println!(
        "interpolated {} columns of length 2^{} into polynomials in {:.2} sec",
        trace.num_cols(),
        log2(trace.num_rows()),
        interpolate_result
    );

    println!(
        "extended {} columns from 2^{} to 2^{} ({}x blowup) in {:.2} sec",
        trace.num_cols(),
        log2(trace.num_rows()),
        log2(extended_trace.num_rows()),
        domain.trace_to_lde_blowup(),
        lde_result
    );

    println!(
        "built Merkle tree from a matrix with {} columns and 2^{} rows in {:.2} sec",
        extended_trace.num_cols(),
        log2(extended_trace.num_rows()),
        mtree_result
    );

    println!("total runtime {:.2} sec", overall_result);
}

// HELPER FUNCTIONS
// ================================================================================================

fn build_domain(num_cols: usize, log_n_rows: u32, blowup_factor: usize) -> StarkDomain<Felt> {
    let num_rows = 2_usize.pow(log_n_rows);
    let trace_info = TraceInfo::new(num_cols, num_rows);
    let options = ProofOptions::new(
        40,
        blowup_factor,
        0,
        HashFunction::Blake3_256,
        FieldExtension::None,
        8,
        64,
    );
    let air = DummyAir::new(trace_info, Felt::ZERO, options);
    StarkDomain::new(&air)
}

#[allow(dead_code)]
fn build_rand_matrix<E: FieldElement>(num_cols: usize, log_n_rows: u32) -> Matrix<E> {
    let mut data = (0..num_cols).map(|_| Vec::new()).collect::<Vec<Vec<E>>>();
    data.par_iter_mut().for_each(|v| {
        *v = rand_vector(2_usize.pow(log_n_rows));
    });

    Matrix::new(data)
}

#[allow(dead_code)]
fn build_fib_matrix<E: FieldElement>(num_cols: usize, log_n_rows: u32) -> Matrix<E> {
    let num_rows = 2_usize.pow(log_n_rows);
    let mut data = (0..num_cols)
        .map(|_| Vec::with_capacity(num_rows))
        .collect::<Vec<Vec<E>>>();

    data.par_iter_mut().enumerate().for_each(|(i, column)| {
        column.push(E::from(i as u64));
        column.push(E::from(i as u64));
        for i in 2..num_rows {
            column.push(column[i - 1] + column[i - 2]);
        }
    });

    Matrix::new(data)
}

// DUMMY AIR
// ================================================================================================

struct DummyAir {
    context: AirContext<Felt>,
}

impl Air for DummyAir {
    type BaseField = Felt;
    type PublicInputs = Felt;

    fn new(trace_info: TraceInfo, _pub_inputs: Felt, options: ProofOptions) -> Self {
        let degrees = vec![TransitionConstraintDegree::new(3)];
        Self {
            context: AirContext::new(trace_info, degrees, 2, options),
        }
    }

    fn evaluate_transition<E: FieldElement<BaseField = Self::BaseField>>(
        &self,
        _frame: &EvaluationFrame<E>,
        _periodic_values: &[E],
        _result: &mut [E],
    ) {
        unimplemented!()
    }

    fn get_assertions(&self) -> Vec<Assertion<Self::BaseField>> {
        unimplemented!()
    }

    fn context(&self) -> &AirContext<Self::BaseField> {
        &self.context
    }
}

// COMMAND LINE OPTIONS
// ================================================================================================

#[derive(StructOpt, Debug)]
#[structopt(name = "sbench", about = "STARK benchmarks")]
pub struct BenchOptions {
    /// Number of columns
    #[structopt(short = "c", long = "columns", default_value = "100")]
    num_cols: usize,

    /// Number of rows expressed as log2.
    #[structopt(short = "n", long = "log_n_rows", default_value = "20")]
    log_n_rows: u32,

    /// Blowup factor, must be a power of two
    #[structopt(short = "b", long = "blowup", default_value = "8")]
    blowup: usize,

    // Field extension degree, must be either 1, 2, or 3
    #[structopt(short = "e", long = "extension", default_value = "1")]
    extension_degree: usize,
}
