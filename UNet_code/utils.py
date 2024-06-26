# https://stackoverflow.com/questions/49595663/find-a-gpu-with-enough-memory
import subprocess
import torch
import logging

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
    result = result.decode() # Convert from binary to string
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def select_cpu_or_gpu():
    device = 'cpu'
    if torch.cuda.is_available():
        # Enable tf32 on Ampere GPUs - only 8% slower than float16 & almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        device = 'cuda:{}'
        gpu_memory = get_gpu_memory_map()
        # Get the index of the GPU with the most available memory, and use that one
        max_gpu = sorted(gpu_memory, key = lambda x: gpu_memory[x])[0]
        device = device.format(max_gpu)
        print('Running on GPU: ' + str(device))
    else:
        print('Warning: model is running on CPU. This may be very slow!')
    return device