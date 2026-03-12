# Changelog

All notable changes to this project will be documented in this file.

This project loosely follows the principles of
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and semantic versioning.

---

## [0.1.0] - Initial development

### Added

- Implementation of the **Truncated Rank Aggregation (TRA)** statistic
- Exact finite-sample null survival evaluation using multinomial / dynamic programming recursion
- Independent ordered-simplex integral backend for validation
- Fixed-\(k\) asymptotic null survival backend
- Fast batched grid evaluation for survival functions
- Rank-wise rejection threshold API
- High-level TRA test interface returning statistic and p-value
- Distribution object (`TRADistribution`) for repeated evaluations
- Functional API:
  - `statistic`
  - `sf`, `sf_grid`
  - `isf`
  - `pvalue`
  - `test`
  - `thresholds`
- Support for multiple evaluation methods:
  - `"exact"`
  - `"simplex"`
  - `"asymptotic"`
- Comprehensive test suite validating agreement between backends
- Repository metadata and documentation:
  - README
  - CITATION.cff