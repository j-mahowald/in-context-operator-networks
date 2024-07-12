#!/bin/bash
set -euo pipefail

# Set environment variables
export JAX_PLATFORMS=cpu

# Set other variables
dir=data
testeqns=100
testquests=5
traineqns=1000

# Print environment for debugging
echo "Environment variables:"
env | sort

# Function to run datagen with common parameters
run_datagen() {
  echo "Running: python3 data_preparation/datagen.py --dir $dir --caption_mode $1 --name $1 --eqns $2 --quests $3 ${*:4}"
  if ! python3 data_preparation/datagen.py --dir "$dir" --caption_mode "$1" --name "$1" --eqns "$2" --quests "$3" "${@:4}"; then
    echo "Error in generating data for ${*:4}" >&2
    return 1
  fi
}

# Run all data generation commands
{
  run_datagen test "$testeqns" "$testquests" --length 100 --dt 0.01 --eqn_types series_damped_oscillator --seed 101 &&
  run_datagen test "$testeqns" "$testquests" --eqn_types ode_auto_const --seed 102 &&
  run_datagen test "$testeqns" "$testquests" --eqn_types ode_auto_linear1 --seed 103 &&
  run_datagen test "$testeqns" "$testquests" --eqn_types ode_auto_linear2 --seed 104 &&
  run_datagen test "$testeqns" "$testquests" --length 100 --dx 0.01 --eqn_types pde_poisson_spatial --seed 105 &&
  run_datagen test "$testeqns" "$testquests" --length 100 --dx 0.01 --eqn_types pde_porous_spatial --seed 106 &&
  run_datagen test "$testeqns" "$testquests" --length 100 --dx 0.01 --eqn_types pde_cubic_spatial --seed 107 &&
  run_datagen test "$testeqns" "$testquests" --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types mfc_gparam_hj --seed 108 &&
  run_datagen test "$testeqns" "$testquests" --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types mfc_rhoparam_hj --seed 109 &&
  run_datagen train "$traineqns" 1 --length 100 --dt 0.01 --eqn_types series_damped_oscillator --seed 1 &&
  run_datagen train "$traineqns" 1 --eqn_types ode_auto_const --seed 2 &&
  run_datagen train "$traineqns" 1 --eqn_types ode_auto_linear1 --seed 3 &&
  run_datagen train "$traineqns" 1 --eqn_types ode_auto_linear2 --seed 4 &&
  run_datagen train "$traineqns" 1 --length 100 --dx 0.01 --eqn_types pde_poisson_spatial --seed 5 &&
  run_datagen train "$traineqns" 1 --length 100 --dx 0.01 --eqn_types pde_porous_spatial --seed 6 &&
  run_datagen train "$traineqns" 1 --length 100 --dx 0.01 --eqn_types pde_cubic_spatial --seed 7 &&
  run_datagen train "$traineqns" 1 --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types mfc_gparam_hj --seed 8 &&
  run_datagen train "$traineqns" 1 --length 100 --dx 0.01 --dt 0.02 --nu_nx_ratio 1 --eqn_types mfc_rhoparam_hj --seed 9 
} || {
  echo "An error occurred during data generation" >&2
  exit 1
}

echo "Data generation completed successfully"
