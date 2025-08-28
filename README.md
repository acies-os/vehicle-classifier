# Acies Vehicle Classifiers

Acoustic- and seismic-based vehicle classifiers.

## Install

```shell
$ git clone git@github.com:acies-os/acies-vehicle-classifier.git
$ cd acies-vehicle-classifier
$ rye sync
```

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

## Usage

There are 4 classifiers, to see their help message:

```shell
$ rye run acies-simple-classifier --help
$ rye run acies-deepsense-classifier --help
$ rye run acies-neusymbolic-classifier --help
$ rye run acies-foundationsense-classifier --help
```

To see debug output, set the environment variable `ACIES_LOG` to desired level:

```shell
$ ACIES_LOG=debug rye run acies-simple-classifier --help
```

