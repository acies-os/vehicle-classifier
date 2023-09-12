# Acies Vehicle Classifiers

Acoustic- and seismic-based vehicle classifiers.

## Install

```shell
$ git clone git@github.com:acies-os/acies-vehicle-classifier.git
$ cd acies-vehicle-classifier
$ rye sync
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
