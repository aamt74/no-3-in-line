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

To use the tool, run the following commands from an **anaconda** prompt:

```
cd $ROOT
conda activate n3ilsat
./n3ilsat --help
```
