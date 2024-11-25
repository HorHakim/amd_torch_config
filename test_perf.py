import torch
import time
from typing import Tuple
from amd_torch_config import configure_torch

def run_performance_test() -> Tuple[float, float, float]:
    """
    Exécute des tests de performance pour vérifier l'efficacité de la configuration.
    
    Returns:
        Tuple[float, float, float]: Temps d'exécution pour (matmul, conv2d, attention)
    """
    if not torch.cuda.is_available():
        return (-1, -1, -1)
    
    # Test 1: Multiplication matricielle
    size = 2048
    a = torch.randn(size, size, device='cuda')
    b = torch.randn(size, size, device='cuda')
    torch.cuda.synchronize()
    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    matmul_time = time.time() - start
    
    # Test 2: Convolution 2D
    batch_size = 32
    conv = torch.nn.Conv2d(3, 64, kernel_size=3).cuda()
    x = torch.randn(batch_size, 3, 224, 224, device='cuda')
    torch.cuda.synchronize()
    start = time.time()
    y = conv(x)
    torch.cuda.synchronize()
    conv_time = time.time() - start
    
    # Test 3: Attention (transformer-style)
    seq_len = 512
    hidden_size = 768
    batch_size = 16
    q = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    k = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    v = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    torch.cuda.synchronize()
    start = time.time()
    attention = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    attention_time = time.time() - start
    
    return matmul_time, conv_time, attention_time

if __name__ == "__main__":
    print("Performance avant configuration:")
    before_times = run_performance_test()
    print(f"MatMul: {before_times[0]:.3f}s")
    print(f"Conv2D: {before_times[1]:.3f}s")
    print(f"Attention: {before_times[2]:.3f}s")
    
    print("\nApplication de la configuration...")
    status = configure_torch()
    
    print("\nPerformance après configuration:")
    after_times = run_performance_test()
    print(f"MatMul: {after_times[0]:.3f}s")
    print(f"Conv2D: {after_times[1]:.3f}s")
    print(f"Attention: {after_times[2]:.3f}s")
    
    print("\nAmélioration des performances:")
    for op, (before, after) in zip(['MatMul', 'Conv2D', 'Attention'], 
                                 zip(before_times, after_times)):
        improvement = (before - after) / before * 100
        print(f"{op}: {improvement:.1f}% d'amélioration")