#!/bin/bash

for i in {8..8}
do
    python evaluating_with_simulator.py --csv_path /app/gearformer_model/ab_SFT_pos_"$i".csv
done
