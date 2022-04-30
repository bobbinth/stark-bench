use prover::{
    crypto::{hashers::Rp64_256, ElementHasher, MerkleTree},
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

pub fn main() {
    let num_cols = 100;
    let log_n_rows = 23;
    let blowup_factor = 4;

    let domain = build_domain(num_cols, log_n_rows, blowup_factor);

    // ----- base field ---------------------------------------------------------------------------

    let result = perform_intt::<Felt>(num_cols, log_n_rows, "base");
    let result = perform_lde(result, &domain, "base");
    let result = build_merkle_tree::<Felt, Rp64_256>(result);
    std::mem::forget(result);

    // ----- quadratic extension ------------------------------------------------------------------

    let result = perform_intt::<QuadExtension<Felt>>(num_cols, log_n_rows, "quad");
    let result = perform_lde(result, &domain, "quad");
    std::mem::forget(result);

    // ----- cubic extension ----------------------------------------------------------------------

    let result = perform_intt::<CubeExtension<Felt>>(num_cols, log_n_rows, "cube");
    let result = perform_lde(result, &domain, "cube");
    std::mem::forget(result);
}

// BENCHMARK FUNCTIONS
// ================================================================================================

fn perform_intt<E: FieldElement>(num_cols: usize, log_n_rows: u32, field: &str) -> Matrix<E> {
    let matrix = build_rand_matrix::<E>(num_cols, log_n_rows);

    let now = Instant::now();
    let result = matrix.interpolate_columns();
    println!(
        "[{}] interpolated {} columns of length 2^{} into polynomials in {:.2} sec",
        field,
        matrix.num_cols(),
        log2(matrix.num_rows()),
        now.elapsed().as_millis() as f64 / 1000_f64
    );
    result
}

fn perform_lde<E: FieldElement>(
    matrix: Matrix<E>,
    domain: &StarkDomain<E::BaseField>,
    field: &str,
) -> Matrix<E> {
    let now = Instant::now();
    let matrix = matrix.interpolate_columns();
    let result = matrix.evaluate_columns_over(domain);
    println!(
        "[{}] extended {} columns from 2^{} to 2^{} ({}x blowup) in {:.2} sec",
        field,
        matrix.num_cols(),
        log2(matrix.num_rows()),
        log2(domain.lde_domain_size()),
        domain.trace_to_lde_blowup(),
        now.elapsed().as_millis() as f64 / 1000_f64
    );
    result
}

fn build_merkle_tree<E: FieldElement, H: ElementHasher<BaseField = E::BaseField>>(
    matrix: Matrix<E>,
) -> MerkleTree<H> {
    let now = Instant::now();
    let result = matrix.commit_to_rows();
    println!(
        "build Merkle tree from a matrix with {} columns and 2^{} rows in {:.2} ms",
        matrix.num_cols(),
        log2(matrix.num_rows()),
        now.elapsed().as_millis() as f64 / 1000_f64
    );
    result
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

fn build_rand_matrix<E: FieldElement>(num_cols: usize, log_n_rows: u32) -> Matrix<E> {
    let mut data = (0..num_cols).map(|_| Vec::new()).collect::<Vec<Vec<E>>>();
    data.par_iter_mut().for_each(|v| {
        *v = rand_vector(2_usize.pow(log_n_rows));
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
