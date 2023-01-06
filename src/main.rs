use miden_crypto::hash::rpo::Rpo256;
use prover::{
    crypto::{hashers::Blake3_256, Digest, ElementHasher},
    iterators::*,
    math::{
        fields::{f64::BaseElement as Felt, CubeExtension, QuadExtension},
        log2, FieldElement,
    },
    Air, AirContext, Assertion, EvaluationFrame, Matrix, ProofOptions, StarkDomain, TraceInfo,
    TransitionConstraintDegree,
};
use rand_utils::rand_vector;
use std::time::Instant;
use structopt::StructOpt;

pub fn main() {
    // read command-line args
    let options = BenchOptions::from_args();

    match (options.hash_fn.as_str(), options.extension_degree) {
        ("blake3", 1) => run_benchmarks::<Felt, Blake3_256<Felt>>(options),
        ("blake3", 2) => run_benchmarks::<QuadExtension<Felt>, Blake3_256<Felt>>(options),
        ("blake3", 3) => run_benchmarks::<CubeExtension<Felt>, Blake3_256<Felt>>(options),
        ("rpo", 1) => run_benchmarks::<Felt, Rpo256>(options),
        ("rpo", 2) => run_benchmarks::<QuadExtension<Felt>, Rpo256>(options),
        ("rpo", 3) => run_benchmarks::<CubeExtension<Felt>, Rpo256>(options),
        _ => unimplemented!(),
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
    //let trace = build_rand_matrix::<E>(options.num_cols, options.log_n_rows);
    let trace = build_fib_matrix::<E>(options.num_cols, options.log_n_rows);
    println!(
        "prepared benchmark inputs in {:.2} sec",
        now.elapsed().as_millis() as f64 / 1000_f64
    );

    // perform interpolation
    let start = Instant::now();
    let polys = trace.interpolate_columns();
    let interpolate_result = start.elapsed().as_millis() as f64 / 1000_f64;

    // perform evaluation
    let eval_start = Instant::now();
    let extended_trace = polys.evaluate_columns_over(&domain);
    let eval_result = eval_start.elapsed().as_millis() as f64 / 1000_f64;
    let lde_result = start.elapsed().as_millis() as f64 / 1000_f64;

    // build Merkle tree
    let mtree_start = Instant::now();
    let tree = extended_trace.commit_to_rows::<H>();
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
        "evaluated {} polynomials of length 2^{} over domain 2^{} in {:.2} sec",
        extended_trace.num_cols(),
        log2(polys.num_rows()),
        log2(extended_trace.num_rows()),
        eval_result,
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
        "built Merkle tree from a matrix with {} columns and 2^{} rows using {} hash function in {:.2} sec",
        extended_trace.num_cols(),
        log2(extended_trace.num_rows()),
        options.hash_fn,
        mtree_result
    );

    println!("Merkle tree root: {}", hex::encode(tree.root().as_bytes()));

    println!("total runtime {overall_result:.2} sec");
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
        prover::HashFunction::Blake3_256,
        prover::FieldExtension::None,
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
        column.push(E::from(i as u64 + 1));
        column.push(E::from(i as u64 + 1));
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
    /// Number of columns.
    #[structopt(short = "c", long = "columns", default_value = "100")]
    num_cols: usize,

    /// Number of rows expressed as log2.
    #[structopt(short = "n", long = "log_n_rows", default_value = "20")]
    log_n_rows: u32,

    /// Blowup factor, must be a power of two.
    #[structopt(short = "b", long = "blowup", default_value = "8")]
    blowup: usize,

    /// Hash function; must be either blake3 or rpo.
    #[structopt(short = "h", long = "hash_fn", default_value = "blake3")]
    hash_fn: String,

    /// Field extension degree, must be either 1, 2, or 3.
    #[structopt(short = "e", long = "extension", default_value = "1")]
    extension_degree: usize,
}
