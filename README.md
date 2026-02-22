# Data Replication

This repository contains the programs used to compute the upper and lower bounds on the optimal consistency reported in the paper for the binary prediction setting.

There are two components:

- `main` — certifies the upper bound for a fixed policy using negative-cycle detection.
- `strategy_iteration` — certifies a lower bound via a mean-payoff game.

---

## Build

Requires a C++20-compatible compiler.

Example:

```bash
g++ -O3 -std=c++20 main.cpp -o main
g++ -O3 -std=c++20 strategy_iteration.cpp -o strategy_iteration
```

## Upper Bound
To cerify the competitive ratio of an algorithm:
```bash
./main
```
The program loads the policy from ``policy-501.csv``. It repeatedly searches for negative cycles. If none are found, the final printed ratio is the certified competitive ratio for the chosen discretization ($\lambda = 500$, where $\Delta = 1/\lambda$). 

## Lower Bound
To certify a lower bound via strategy iteration:
```bash
./strategy_iteration
```
Provide parameters via standard input:
```code
p
q
```
The competitive ratio is $c = p / q$. The following example
```code
7747
5000
```
certifies a lower bound of $7747/5000 = 1.5494$. 
