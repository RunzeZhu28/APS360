import torch

if torch.cuda.is_available():
    print("GPU可用")

    current_gpu_device = torch.cuda.current_device()
    print("当前正在使用的GPU设备编号:", current_gpu_device)

    gpu_name = torch.cuda.get_device_name(current_gpu_device)
    print("GPU设备名称:", gpu_name)

    gpu_total_memory = torch.cuda.get_device_properties(current_gpu_device).total_memory
    gpu_available_memory = torch.cuda.get_device_properties(current_gpu_device).total_memory
    print("GPU总内存量:", gpu_total_memory)
    print("GPU可用内存量:", gpu_available_memory)
else:
    print("GPU不可用")
