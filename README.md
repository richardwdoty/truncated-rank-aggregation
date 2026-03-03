# Truncated Rank Aggregation (TRA)

Python implementation of **Truncated Rank Aggregation (TRA)**, a statistical
procedure for combining the most informative order statistics among a
collection of p-values.

This repository provides efficient evaluation of the null survival function

S_{n:k}(c) = P(T_{n:k} > c)

including:

- exact finite-sample evaluation via multinomial occupancy recursion
- ordered-simplex integral validation
- fixed-k asymptotic approximations

The method forms the statistical core of the **FRACTEL** framework for CRISPR
perturbation screens.

## Installation

```bash
pip install truncated-rank-aggregation
````

or for development

```bash
pip install -e ".[dev]"
```

## Example

```python
import tra

# survival probability
tra.sf(c=0.05, n=100, k=5)
```