#!/bin/bash

# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <integer>"
    exit 1
fi

# Check if the provided argument is an integer
if ! [[ $1 =~ ^[0-9]+$ ]]; then
    echo "Error: Argument must be an integer"
    exit 1
fi

# Loop from 0 to the input integer
for (( i=0; i<=$1 - 1; i++ )); do
    # Run the Python file with i as the input argument
    python3 generate.py "--cpu_num" "$i" "--num_of_cpus" "$1" &
done

wait