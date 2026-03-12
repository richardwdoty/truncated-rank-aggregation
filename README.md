# Truncated Rank Aggregation (TRA)

**Truncated Rank Aggregation (TRA)** is a statistical procedure for combining the most informative order statistics from a collection of p-values. The method evaluates the statistic

\[
T_{n:k} = \min_{1 \le i \le k} R_i,
\]

where \(R_i\) is the probability–integral transform (PIT) of the \(i\)-th order statistic under the global null.

This repository provides a Python implementation of TRA, including:

- exact finite-sample null survival evaluation via a multinomial / dynamic programming recursion
- an independent ordered-simplex integral representation for validation
- fixed-\(k\) asymptotic approximations via a Poisson-process limit
- fast grid evaluation for plotting and calibration
- rank-wise rejection thresholds for interpretability

TRA is the statistical core underlying the **FRACTEL** framework for CRISPR perturbation screens, but the method itself is fully general and applies to any setting where ordered p-values are aggregated.

---

## Installation

```bash
pip install truncated-rank-aggregation
````

For development:

```bash
pip install -e .
```

---

## Quick Example

```python
import numpy as np
import tra

# generate example p-values
pvals = np.random.uniform(size=100)

# run the TRA test
result = tra.test(pvals, k=5)

print(result.statistic)
print(result.pvalue)
```

---

## Computing the statistic

```python
import tra

t = tra.statistic(pvals, k=5)
```

This computes

[
T_{n:k} = \min_{1 \le i \le k} R_i.
]

---

## Computing p-values

```python
p = tra.pvalue(pvals, k=5)
```

Internally this evaluates the null survival function

[
S_{n:k}(c) = P(T_{n:k} > c).
]

---

## Null survival functions

Evaluate the null survival function at a single point:

```python
tra.sf(c=0.05, n=100, k=5)
```

or over a grid:

```python
import numpy as np

grid = np.linspace(0, 1, 200)
S = tra.sf_grid(grid, n=100, k=5)
```

Available backends:

* `"exact"` — finite-sample DP recursion
* `"simplex"` — ordered-simplex integral representation
* `"asymptotic"` — fixed-(k) Poisson-process limit

Example:

```python
tra.sf(c=0.05, n=100, k=5, method="simplex")
tra.sf(c=0.05, k=5, method="asymptotic")
```

---

## Rejection thresholds

TRA can also be expressed as rank-wise rejection thresholds:

```python
a = tra.thresholds(alpha=0.05, n=100, k=5)
```

The test rejects if

[
P_{(i)} \le a_i
]

for some (i \le k).

These thresholds often provide a more interpretable view of the test than the statistic itself.

---

## Distribution object

For repeated evaluations, you can construct a distribution object:

```python
dist = tra.null_dist(n=100, k=5)

dist.sf(0.1)
dist.sf_grid(np.linspace(0,1,100))
dist.isf(0.05)
dist.thresholds(0.05)
dist.test(pvals)
```

For the fixed-(k) asymptotic limit:

```python
dist = tra.null_dist(k=5, method="asymptotic")
dist.sf(0.1)
```

---

## Features

* exact finite-sample null distribution
* independent validation via simplex integration
* fast batched grid evaluation
* asymptotic Poisson-process limit
* rank-wise rejection thresholds
* clean statistical API

---

## Citation

If you use this package in research, please cite the accompanying paper on **Truncated Rank Aggregation**.

A `CITATION.cff` file will be provided for automated citation metadata.

---

## License

This project is distributed under the terms of the repository license.