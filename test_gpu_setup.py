#!/usr/bin/env python3
import os
import torch

def test_gpu_setup():
    print("=== Test GPU Setup ===")
    
    # Test 1: Trước khi set CUDA_VISIBLE_DEVICES
    print("\n1. Trước khi set CUDA_VISIBLE_DEVICES:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Total GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test 2: Set CUDA_VISIBLE_DEVICES=1
    print("\n2. Sau khi set CUDA_VISIBLE_DEVICES=1:")
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    
    # Reload torch để nhận env vars mới
    import importlib
    importlib.reload(torch)
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Visible GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
    
    # Test 3: Tạo tensor trên GPU
    if torch.cuda.is_available():
        print("\n3. Test tạo tensor trên GPU:")
        device = torch.device("cuda:0")
        x = torch.randn(3, 3).to(device)
        print(f"Tensor device: {x.device}")
        print(f"Tensor shape: {x.shape}")
        print("✓ GPU test thành công!")
    else:
        print("\n3. Không có GPU available")

if __name__ == "__main__":
    test_gpu_setup()
