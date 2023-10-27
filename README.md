

# Representing Edge Flows on Graphs via Sparse Cell Complexes

<img align="right" width="200" style="margin-top:-5px" src="readme_src/LOGO_ERC-FLAG_FP.png">

[![arXiv:2309.01632](https://img.shields.io/badge/arXiv-2309.01632-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2309.01632)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/josefhoppe/edge-flow-cell-complexes/blob/main/LICENSE)
![uses Conda](https://img.shields.io/badge/Conda-44A833?logo=Anaconda&logoColor=white)
[![Snakemake 7.3 or higher](https://img.shields.io/badge/Snakemake-≥7.3.0-039475.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAABhGlDQ1BJQ0MgcHJvZmlsZQAAKJF9kT1Iw0AcxV9TRSlVh3aQ4hCwOlkQFXHUKhShQqkVWnUwufQLmjQkKS6OgmvBwY/FqoOLs64OroIg+AHi6uKk6CIl/i8ptIjx4Lgf7+497t4BQqPCVLNrHFA1y0gn4mI2tyr2vCKACPoRwrDETH0ulUrCc3zdw8fXuxjP8j735+hT8iYDfCLxLNMNi3iDeHrT0jnvE4dZSVKIz4nHDLog8SPXZZffOBcdFnhm2Mik54nDxGKxg+UOZiVDJZ4ijiqqRvlC1mWF8xZntVJjrXvyFwbz2soy12kOIYFFLCEFETJqKKMCCzFaNVJMpGk/7uGPOP4UuWRylcHIsYAqVEiOH/wPfndrFiYn3KRgHOh+se2PEaBnF2jWbfv72LabJ4D/GbjS2v5qA5j5JL3e1qJHwMA2cHHd1uQ94HIHGHzSJUNyJD9NoVAA3s/om3JA6BYIrLm9tfZx+gBkqKvkDXBwCIwWKXvd4929nb39e6bV3w91xnKo0GhamgAAAAZiS0dEAA4ADgAOq1UFEwAAAAlwSFlzAAAWJQAAFiUBSVIk8AAAAAd0SU1FB+cKGwwkI1CoIBUAAAKYSURBVEjHtZZPSFRxEMc/897z76oXrUNQYUVBXoIINimiQ0T/IAo6JAQhZeiTKHrmJehW7IoQ7m4aHSoIO5h1qoOBhlGJ0SUjiqggAskizQxc33vTwbexrouYu83tzczv953fzHdmnpAhZsypBU4C24FVgBmYhOyiQBJ4Dwwg0uU1RV6lO0ja5WVAJ3B0gQsXIz5wBeW81xydIRWllXBCKI+AvUsBuLx+C90H6qmYnKT/x6gAWxE2yJ5wrz58pgaA+nQA4aWGXl5UQkVpiFBhSbr6iIh5FkDMmLMZGM4xRVQXFPFxZjpTPQVSbQANuQIA2QAAQojWGMBO/p+8BBm2gDVZjCNAAhhCdGqWp8aCDxY80v0EdU1xPyQb29XKcrI9oJ9rdThlqlI+e4kGLZENwPjlNrdNZuq9lN2MOZ+A1cF3t6FS54tuBK4C2/6hXqNAHxD37OjQnCDMmHMTOAbMIFINJFEdAZYvsQ4KXMOwTnuNl6YBDOB6YHzhNUW+oHo8B4DUFGnAd3usWIsJYHh2dBDoBT4HTuvyxKz9iramXoIg9QGjAN7lkcKtZtyplPlT+FwVyOscU5Yup4x5tLPbvgUNOhBM1FyldkF6FnQ6hSZanHWeq2H4vpiiWuCjVcAO4AywNsP1iQCY8ZYQqisXxRyR2SWm6qYpv7t2ZMxMOKX43AIOp50ZFDPu1KA8BipzSIkCD4B6DMbxeQ5sCmw3DJSDOQKkemMfcF+UJHAxPV0W8DSPlA0rhBHpQ9UDfgrSY3h2tB+4mzcYZYXXFPkNjAMXXDsykaLwCeBNnmBSzdxFMYk5fytWzFmmcBvYlQPAPc+OHspU/m1G146OgewOCngHeAtMAG6WRZJaLh4wA3wFOhCjLhvyH6jS1OKPekJ4AAAAAElFTkSuQmCC)](https://snakemake.readthedocs.io)


## About

This repository accompanies our paper linked above.
The code is split into two repositories:

- This repository contains the evaluation code and snakemake workflow.
- [Cell FLOWer](https://github.com/josefhoppe/cell-flower) is a separate package that contains only the code required for the inference and is published to PyPI.

```
@misc{hoppe2023representing,
      title={Representing Edge Flows on Graphs via Sparse Cell Complexes}, 
      author={Josef Hoppe and Michael T. Schaub},
      year={2023},
      eprint={2309.01632},
      archivePrefix={arXiv},
      primaryClass={cs.SI}
}
```

## Running numerical experiments

This project uses [Snakemake](https://snakemake.readthedocs.io).
To run the experiments:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or mamba)
2. Create a new conda environment and install dependencies: `conda env create -n cell-inference -f workflow/envs/environment.yaml`
3. Activate environment: `conda activate cell-inference`
4. Run snakemake: `snakemake -c20 all`

For us, running all experiments takes approximately three hours, so you may want to build individual artifacts instead, for example `snakemake -c20 figures/approx_exp_error.pdf`.

## Workflow structure

The Rulegraph gives a rough overview of the structure:

![Snakemake Rulegraph](readme_src/rules.png)

Generally, we aimed to give rules names that are short and easily understood.
For the different kinds of experiments, we used the following naming system:

- **approx_exp** - approximation experiment on synthetic data, i.e., how large is the approximation error?
- **inference_exp** - inference experiment on synthetic data, i.e., how many ground-truth cells have been recovered?
- **realworld_exp** - approximation experiment on realworld-data.
- **time_exp** - experiment to measure runtime performance.
- **heuristic_exp** - experiment to see how many ground-truth cells are detected by the heuristic (first iteration).

## Acknowledgements

Funded by the European Union (ERC, HIGH-HOPeS, 101039827). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.