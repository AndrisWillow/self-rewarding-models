# This directory contains all of the benchmarks and evaluations for this project

## run-all
This script will execute all of the benchmarks in this directory except for
IFEval and alpaca-eval that need to be evaluated externally.
More info about them in their respective directories

## To launch the script
Launch from /app/benchmarks directory:

./run-all.sh BASE_MODEL RESULT_NAME ADAPTER_PATH

## Example scipt usages that corresond to the results in my paper:

## Experiment 1:
### Mistral-7B-Instruct-v0.2-M0:
./run-all.sh mistralai/Mistral-7B-Instruct-v0.2 Mistral-7B-Instruct-v0.2-M0

### Mistral-7B-Instruct-v0.2-M1:
./run-all.sh mistralai/Mistral-7B-Instruct-v0.2 Mistral-7B-Instruct-v0.2-M1 Mistral-7B-Instruct-v0.2-SFT_baseline_IFT+EFT

### Mistral-7B-Instruct-v0.2-M2:
./run-all.sh mistralai/Mistral-7B-Instruct-v0.2 Mistral-7B-Instruct-v0.2-M2 Mistral-7B-Instruct-v0.2-SFT_baseline-DPO-M2

## Experiment 2:
### Mistral-7B-v0.1-M0:
./run-all.sh mistralai/Mistral-7B-v0.1 Mistral-7B-v0.1-M0