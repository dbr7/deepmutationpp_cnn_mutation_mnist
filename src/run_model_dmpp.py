import os, sys
import datetime
print(datetime.datetime.now())

import subprocess


# model_path = os.path.join('..', '..', 'original_models', 'mnist_original_0.h5')
# mutants_path = os.path.join('..', '..', 'mutated_models_mnist')

model_no = sys.argv[1]
gpu_no = int(model_no) % 4
model_path = f'/home/DC_replication/DCReplication/deepmutationpp/original_models/multiple_training/mnist_original_{model_no}.h5'
mutants_path = f'/home/DC_replication/DCReplication/deepmutationpp/mutated_models_mnist_trained:{model_no}'

if not os.path.exists(mutants_path):
    os.makedirs(mutants_path)

operators = [0, 1, 2, 3, 4, 5, 6, 7]
# operators = [1, 3, 6]
# ratio_vals = [0.01, 0.03, 0.05]
ratio = 0.05
# num = 400
num = 100
standard_deviation=1.0

for operator in operators:
    bash_command = f"CUDA_VISIBLE_DEVICES={gpu_no} python generator.py -model_path {model_path} -data_type mnist -operator {operator} -ratio {ratio} -save_path {mutants_path} -num {num} -standard_deviation {standard_deviation}"
    print(bash_command)
    subprocess.call(bash_command, shell=True)
        
    # print(" ".join([f"CUDA_VISIBLE_DEVICES={gpu_no}", "python", "generator.py",
    #                 "-model_path", model_path,
    #                 "-data_type", "mnist",
    #                 "-operator", str(operator),
    #                 "-ratio", str(ratio),
    #                 "-save_path", mutants_path,
    #                 "-num", str(num)]))

print(datetime.datetime.now())