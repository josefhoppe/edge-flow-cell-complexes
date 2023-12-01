

# Representing Edge Flows on Graphs via Sparse Cell Complexes

<img align="right" width="200" style="margin-top:-5px" src="readme_src/LOGO_ERC-FLAG_FP.png">

[![Published at Learning on Graphs Conference 2023](https://img.shields.io/badge/Published_at-LoG_2023-003380.svg)](https://openreview.net/forum?id=qix189lq5D)
[![Best Paper at LoG 2023 - LoG Conference on Twitter](https://img.shields.io/badge/LoG_2023:_Best_Paper-003380.svg?logo=data:image/png;base64,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)](https://twitter.com/LogConference/status/1729167588851143023)
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
@inproceedings{
      hoppe2023representing,
      title={Representing Edge Flows on Graphs via Sparse Cell Complexes},
      author={Josef Hoppe and Michael T Schaub},
      booktitle={The Second Learning on Graphs Conference},
      year={2023},
      url={https://openreview.net/forum?id=qix189lq5D}
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
