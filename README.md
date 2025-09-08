# Acies Vehicle Classifiers

Acoustic- and seismic-based vehicle classifiers.

## Setup

### Python Environment Management

This project uses [`uv`](https://docs.astral.sh/uv) (or its predecessor [`rye`](https://rye.astral.sh)) to manage the Python environment.

To check if `rye` is already installed, run:

```bash
which rye
```

- If the command prints a path, `rye` is available and you can skip this step.
- If not, we recommend installing `uv` for new setups. Follow [uv’s official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

For backward compatibility, you may also install [rye](https://rye.astral.sh/guide/installation/).

### Clone and install dependencies

```shell
$ git@github.com:acies-os/vehicle-classifier.git
$ cd vehicle-classifier
vehicle-classifier$ uv sync
 
# or, if using rye
vehicle-classifier$ rye sync
```

### Install `just`

Install `just` use [your package manager](https://just.systems/man/en/packages.html) or [pre-built binary](https://just.systems/man/en/pre-built-binaries.html).

## Documentation

The documentation is managed using **Sphinx**, which fetch docstring comments from code and compile them into html pages.

Sphinx sources live in the `docs/` folder:

- `conf.py` — Sphinx configuration
- `*.rst` files — reStructuredText source documents that define docs content

To build Sphinx documentation, run this command:

```shell
$ just build-doc
```

Now the built documentation should live in `build/` folder.

To view the documentation in browser, run this command:

```shell
$ just view-doc
```

## Download Model Weights

To download model weights, either download the `.pt` files yourself from github release page to `models/` folder, or use `wget` to download automatically:

```bash
# in root folder
vehicle-classifier$ cd models/

vehicle-classifier/models$ wget https://github.com/acies-os/vehicle-classifier/releases/download/weight-v1.0.0/gcq202410_mae.pt

vehicle-classifier/models$ wget https://github.com/acies-os/vehicle-classifier/releases/download/weight-v1.0.0/Parkland_TransformerV4_vehicle_classification_finetune_gcq202410_1.0_multiclasslatest.pt
```

## Usage

There are 2 classifiers, to run them:

```shell
$ just vfm
$ just mae
```

To see debug output, set the environment variable `ACIES_LOG` to desired level:

```shell
$ ACIES_LOG=debug rye run acies-simple-classifier --help
```

