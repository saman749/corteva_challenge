# Corteva Surplus unit allocation problem.

This repository contains the implementation of Corteva's surplus unit 
allocation problem, done by Saman Cyrus.

The code is located inside `main.py` file. The index file for the webpage of this project is located at `index.html`. There are three methods that
are tried to solve this problem:
- Robust optimization method using `pulp`.
- Robust optimization method using `pyomo`.
- Two-stage stochastic programming method using `mpi-sppy`.
- Two-stage stochastic programming method using `mpi-sppy` with substitution.

The results are stored in a dataframe called `out` for different values of
`max_capacity`, `macro`, and `sub`.

**max_capacity:** To replace `nan` values in `Capacity` column, we've considered
a variable called `max_capacity`.

**macro:** This is the Macro target percentage, defined in the problem
description as an adjustable input parameter, ranging between 10% and
50%.

**sub:** The value of Substitutability limit. This is the last constraint of 
the problem and is defined to ensure adequate totat surplus quantity. 

**VERBOSE:** Verbosity of the objective function for each run of each algorithm.
The deafult value is 0.

**EPSILON:** Robustness parameter for `pyomo` robust optimization formulation.

**mpisppy_options:** solver of choice for `mpisppy`. The default value is `glpk`.
Other possibilities are `CPLEX`, `Gurobi`, and `CBC` for LP and MILP problems.
Here we've solved the extensive form of the problem directly. Alternatively,
we can use progressive hedging algorithm (choose solver to be `PH` and make 
the `ph_object`).

**mpisppy_all_scenario_names:** Possible scenarios for `mpisppy` two-stage 
stochastic programming package.

## Dependencies

This implementation uses Python, with Conda dependencies specified in 
`environment.yml` and PyPI dependencies specified in `requirements.txt`. You
can create the Conda environment via

    conda env create

or 

    pip install -r requirements.txt

## Unit Tests (didn't finish this part)

Unit tests are implemented via [pytest][pytest].

### Code Coverage

To assess coverage of the codebase by unit tests, we use the `pytest-cov`
plugin which is the interface of `pytest` to [Coverage.py][coveragepy]. To get
a coverage report, run
```
pytest test/core --cov
```
This command will create a summary table in the console. For more details run
```
pytest test/core --cov --cov-report html
```
which will produce HTML files in `test/.coverage_report`.


[pytest]: https://docs.pytest.org/en/latest/contents.html
[coveragepy]: https://coverage.readthedocs.io/en/latest/index.html
