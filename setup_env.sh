# source this file from the repository root:  source setup_env.sh
export SITESHIFT_DATA="${SITESHIFT_DATA:-$PWD/datasets}"
export SITESHIFT_RAW="${SITESHIFT_RAW:-$PWD/raw_downloads}"
export SITESHIFT_RUNS="${SITESHIFT_RUNS:-$PWD/runs}"
export SITESHIFT_PRED="${SITESHIFT_PRED:-$PWD/predictions}"
export SITESHIFT_RESULTS="${SITESHIFT_RESULTS:-$PWD/results}"
export SITESHIFT_OUT="${SITESHIFT_OUT:-$PWD/results/analysis}"
export SITESHIFT_INDEX="${SITESHIFT_INDEX:-$PWD/data}"
export PYTHONPATH="$PWD:$PWD/data:$PWD/train:$PWD/eval:$PWD/figures:${PYTHONPATH:-}"
