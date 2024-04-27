import subprocess

command = ["nvidia-smi"]

try:
    result = subprocess.run(command, capture_output=True, text=True)
    gpu_info = result.stdout
    if gpu_info.find('failed') >= 0:
      print('Not connected to a GPU')
    else:
      print(gpu_info)
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")

import torch 

print(torch.cuda.current_device())