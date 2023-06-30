### Important Files and Folders (in src folder)
- `mnist_conv_train.py`: The original MNIST model training file.
- `deepcrime_scripts`: It contains original DeepCrime scripts such as utils.py or stats.py
- `cnn_operator.py` and `generator.py`: These files are the core of DeepMutation++. I only added few lines of code to set the random seed.
- `run_op_dmpp.py`: This is the main file to call `generator.py`. After generating DeepMutation++ mutatns, it checkes the accuracies of generated mutants to determine the killing of them. *The definition of the killing here is from DeepCrime paper.*
- `compute_ki.py`: This computes the killing infomation for DeepMutation++ mutants. `get_prediction_array_mutant` and `save_killing_info` functions do this job.
- `0_subsumption_study.ipynb`: The jupyter notebook to analyse the redundancy between mutants. There are two experiments: 1) whether DeepCrime mutants subsume DeepMutation++ mutants, 2) whether DeepMutation++ mutants subsume DeepCrime mutants.

