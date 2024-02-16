#!/bin/bash

dataset_versions=("subset_balanced")
# dataset_versions=("subset_balanced" "overall_balanced" "not_balanced" "full_data")

for dataset_version in "${dataset_versions[@]}"; do
    for file in ../inference_results/${dataset_version}/*; do
        if [ -f "$file" ]; then
            # Skip files that contain "analysis" in their name
            if [[ $file == *"analysis"* ]]; then
                continue
            fi

            # Set method based on file prefix
            if [[ $file == *"attrbench"* ]]; then
                method="attrbench"
            elif [[ $file == *"autoais"* ]]; then
                method="autoais"
            else
                # If file does not start with "attrbench" or "autoais", skip the file
                continue
            fi
            
            echo "Processing $file with method $method"
            python ../analysis_inference_results.py --data_path "$file" --method $method
        fi
    done
done
