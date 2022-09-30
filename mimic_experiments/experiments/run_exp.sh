#!/bin/bash
python3 -m mimic_experiments.run_test_filtered_dpsgd_compare_loss 0.5
python3 -m mimic_experiments.run_test_filtered_dpsgd_compare_loss 1
python3 -m mimic_experiments.run_test_filtered_dpsgd_compare_loss 1.5
python3 -m mimic_experiments.run_test_filtered_dpsgd_compare_loss 3
python3 -m mimic_experiments.run_test_filtered_dpsgd_compare_loss 5