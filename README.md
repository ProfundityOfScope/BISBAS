# BISBAS
---

## Introduction 
A GPU accelerated (via [Bifrost](https://github.com/ledatelescope/bifrost)) version of an existing Intermittent Sparse Baseline Subset ([ISBAS](https://github.com/ericlindsey/isbas), originally written by Dr. Eric Lindsey) code used for processing Interferometric Synthetic Aperture Radar (InSAR) datasets. The original scripts this code is based on were created by Dr. Eric Lindsey, with changes made to the algorythm to benefit from GPU parallelization. Current tests show significant speedups. As an example, a test case with 500 interferograms from 170 dates, each about 4500x4500 processed in about 18 minutes, having originally taken something like 80 hours. Exact scaling will depend on image size, number of interferograms, and number of dates.

## Usage
The primary script here is `bisbas.py`, which parses the data and configuration file, then sets up Bifrost blocks (which are defined in `bisblocks.py`) for the run. The actual process is broken into two pipelines, which run one after the other. The first pipeline models and generates the timeseries, and accumulates the matrices which are used to solve for image-level systematic offsets after this pipeline is finished. The second pipeline applies the solution to remove these artificial ramps or bowls, and fits for the velocity of each pixel. 

In its current state the code expects a lot of things, generalization toward more diverse datasets is still a TODO. For now, you'll need an HDF5 file, in a particular format, and you'll need a config file like I've included.
