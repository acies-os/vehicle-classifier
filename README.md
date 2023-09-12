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
$ rye run acies-simple-classifier -h
$ rye run acies-deepsense-classifier -h
$ rye run acies-neusymbolic-classifier -h
$ rye run acies-foundationsense-classifier -h
```

To see debug output, set the environment variable `ACIES_LOG` to desired level:

```shell
$ ACIES_LOG=debug rye run acies-simple-classifier -h
```
