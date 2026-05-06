# Data Replication

Code accompanying the binary prediction section of the paper. Two programs are provided, both operating in the perfect-prediction regime:

- **Upper-bound program** (`upper-bound-program/`): given an online algorithm, certifies a competitive ratio via Bellman–Ford, or returns an adversarial counter-example.
- **Lower-bound program** (`lower-bound-program/`, `lower-bound-program-y-0/`): solves the algorithm-vs-adversary mean-payoff game by strategy iteration, producing an optimal Max-player policy.

Both programs discretize time to multiples of Δ = 1/N for an integer N ≥ 1; larger N gives finer discretization and tighter bounds.

## Build

Requires `g++` with C++20 support. OpenMP is used for multi-threading. Run `make` inside each program's directory.

## Upper-bound program

```bash
cd upper-bound-program
make                    # build ./solver with N=100 (default)
make LAMBDA=1000        # build ./solver with N=1000
./solver [policy.csv] [--csv-grid n] [--bound-mode]
```

| Argument | Meaning |
|----------|---------|
| `policy.csv` | (optional) policy produced by the lower-bound program. If omitted, defaults to the 5/3-consistent algorithm of Zuo et al. (2024). |
| `--csv-grid n` | Discretization N used by the lower-bound program when producing `policy.csv`. Must match that run's N. (default: 50) |
| `--bound-mode` | Use adjusted transfer costs (1+Δ for ALG, 1−Δ for OPT), corresponding to the *adjusted upper-bound program* of the paper. Required for a valid bound in the continuous-time model. |
| `-o <file>` | Write the Bellman–Ford vertex potential to `<file>` (CSV). |

`LAMBDA` is a compile-time constant controlling the size of the state graph; `--csv-grid` is a run-time argument describing the input policy.

The solver iteratively searches for negative cycles, lowering the candidate ratio until none remains. Sample output:

```
Bellman-Ford using 22 thread(s)
Negative cycle found:
The ratio is 5645/3643 = 1.54955
...
No negative cycle found.
The ratio is 8456/5451 = 1.55127
CycleStep:
  ...
```

The final ratio (here 8456/5451 ≈ 1.55127) is the certified competitive ratio. The CycleStep dump shows the tightest cycle found, useful for inspecting the worst-case instance.

## Lower-bound program

Two versions are provided:

- **`lower-bound-program/`** solves the full game described in the paper. **Use this version to compute a lower bound on the optimal consistency.**
- **`lower-bound-program-y-0/`** solves a variant in which the algorithm is restricted, in LTP states, to y = 0. Empirically the two versions yield the same game value, but only this restricted version produces a policy in the format consumed by the upper-bound program. **Use this version to obtain `policy.csv` for upper-bound certification.**

Build and run inside either directory:

```bash
make
./mpg_solver <N> <p> <q> [--verbose]
./mpg_solver 100 3 2 --verbose      # example
```

`N` is the discretization (Δ = 1/N) and `p/q` is the initial candidate ratio. The solver outputs the optimal Max policy as `policy.csv` and the threshold ratio `p*/q*`. With `--verbose`:

```
N=100  initial p/q=3/2  nodes=364005
iter 1  p/q=3/2  g=-49/3
...
iter 14  p/q=82/53  g=0/1
Saved scaled Max policy CSV to policy.csv
p*/q* = 82/53
Certificate verified.
```

The final line confirms that the produced policies and potentials have been verified to satisfy the optimality inequalities described in the paper.

## Reproducing the paper's bounds

**Lower bound** (full game, Δ = 1/1000):
```bash
cd lower-bound-program
make
./mpg_solver 1000 3 2
```

**Upper bound** (adjusted costs, Δ = 1/5000), using a policy from the y=0 version:
```bash
cd lower-bound-program-y-0
make
./mpg_solver 1000 3 2          # produces policy.csv

cd ../upper-bound-program
make LAMBDA=5000
./solver ../lower-bound-program-y-0/policy.csv --csv-grid 1000 --bound-mode -o bf-potential.csv
```

The policy used by the paper is `lower-bound-program-y-0/policy-1000.csv`. Upon obtaining a Bellman-Ford vertex potential, its smoothness constant `L` can be obtained by running `upper-bound-program/findL.py`.
