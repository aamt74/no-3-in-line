# n3ilsat

A small tool that encodes the no-three-in-line problem as a SAT problem, which can then be solved using an external SAT-solver, e.g. glucose or z3. If a satisfiable model has been found by the SAT-solver, the tool can decode it, verify that it indeed is a valid solution to the no-three-in-line problem. Optionally, a nice png can be generated to visualize the solution.

## Setup

We make use of [anaconda](https://www.anaconda.com/products/distribution#Downloads). Assuming the current working directory is ```$ROOT``` where you cloned the repo to, the following commands are to be used from an **anaconda** prompt:

- Create environment:
   ```
   cd $ROOT
   conda env create -n n3ilsat --file environment.yml
   conda activate n3ilsat
   ```
- Update environment:
   ```
   cd $ROOT
   git clean pull
   conda env update --name n3ilsat --file environment.yml --prune
   conda activate n3ilsat
   ```
- Delete environment:
   ```
   cd $ROOT
   conda deactivate
   conda env remove --name n3ilsat
   ```

## Running

To use the tool, perform the following steps from an **anaconda** environment:

- Activate environment:
  ```
  cd $ROOT
  conda activate n3ilsat
  ```
- Encode a 'no-three-in-line' problem of some dimension and symmetry. The
  following statements generate problems for N=10 and N=20 in 'iden' and 'rot4'
  symmetry, respectively: 
  ```
  ./n3ilsat.py encode 10 . > problem.dimacs
  ./n3ilsat.py encode 20 o > problem.dimacs
  ```
- Use an external SAT-solver to solve the problem. Here's an example of how to
  use [glucose](https://github.com/audemard/glucose):
  ```
  glucose -model problem.dimacs solution.model
  ```
- Decode the model found by the external SAT-solver to verify it is indeed a
  correct solution and, optionally, generate a nice png image. Make sure to
  use the same parameters as you did when generating the problem. For example:
  ```
  cat solution.model | ./n3ilsat.py decode 20 o --png
  ```
