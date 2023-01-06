# stark-bench
This repo contains code for benchmarking certain functions from the [Winterfell](https://github.com/novifinancial/winterfell) project. The current list of functions includes:

* Polynomial interpolation using iNTT method.
* Polynomial evaluation using NTT method.
* Low degree extension, which consists of iNTT followed by an NTT over a larger domain.
* Merkle tree construction.

## Usage
To run the benchmarks you'll need to first compile the code in release mode. This can be done like so:
```
cargo build --release
```
The above assumes that you have cloned the repo and installed Rust on your machine.

To view instructions on how to run benchmarks you can execute the following command:
```
./target/release/stark-bench --help
```
Running the above command should print out the following text:
```
USAGE:
    stark-bench [OPTIONS]

FLAGS:
        --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -b, --blowup <blowup>                 Blowup factor, must be a power of two [default: 8]
    -e, --extension <extension-degree>    Field extension degree, must be either 1, 2, or 3 [default: 1]
    -h, --hash_fn <hash-fn>               Hash function; must be either blake3 or rpo [default: blake3]
    -n, --log_n_rows <log-n-rows>         Number of rows expressed as log2 [default: 20]
    -c, --columns <num-cols>              Number of columns [default: 100]
```
Thus, for example, to run the benchmark for an input matrix of 100 columns and 2^23 rows, and perform an LDE with with a blowup factor of 8, you can execute the following command:
```
./target/release/stark-bench -c 100 -n 23 -b 8
```

### Input data sets
Inputs for the benchmarks can be generated using two methodologies:
1. Deterministic inputs based on Fibonacci sequence (this is the default).
2. Random values.

In both cases the results should be similar, but random inputs take considerably more time to generate.

To change the input method, you'll need to update the [main.rs](https://github.com/bobbinth/stark-bench/blob/main/src/main.rs#L37) and recompile the code.

## License
This project is [MIT licensed](./LICENSE).
