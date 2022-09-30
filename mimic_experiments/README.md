# MIMIC-III experiments
1. Obtain access to [MIMIC-III database](https://physionet.org/content/mimiciii/1.4/).
2. Install the enviroment with requirements.txt 
3. Run the preprocessing for the phenotyping task from the benchmark library by [Harutyunyan et al. (2019)](https://doi.org/10.1038/s41597-019-0103-9). The code for that can be found on [GitHub](https://github.com/YerevaNN/mimic3-benchmarks) and save the respective train and test datasets to numpy files.
4. Select subgroups using subgroups/pick_subgroups_non_overlapping.py and subgroups/pick_subgroups_non_overlapping_test_set.py
5. Run the contents of experiments/run_exp.sh from this folder. The losses and gradient norms will be saved to respective folders.
6. Compute the individual epsilons using the respective function in compute_eps/eps.py