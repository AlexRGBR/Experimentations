import time

import torch
import torch.nn as nn


def evaluate_module_time(
    module: nn.Module,
    input_tensors: list[torch.Tensor],
    n_iterations: int = 50,
    n_dry_runs: int = 10,
) -> tuple[float, list[float]]:
    """Time `module(*input_tensors)` over several iterations.

    Runs `n_dry_runs` untimed forward passes first (JIT/cudnn warm-up,
    lazy allocations, ...) so the timed iterations reflect steady-state
    performance. Returns (total_time_seconds, per_iteration_times_seconds).
    """
    device = input_tensors[0].device
    is_cuda = device.type == "cuda"

    module.eval()
    with torch.no_grad():
        for _ in range(n_dry_runs):
            module(*input_tensors)
        if is_cuda:
            torch.cuda.synchronize(device)

        iteration_times = []
        start = time.perf_counter()
        for _ in range(n_iterations):
            iteration_start = time.perf_counter()
            module(*input_tensors)
            if is_cuda:
                torch.cuda.synchronize(device)
            iteration_times.append(time.perf_counter() - iteration_start)
        total_time = time.perf_counter() - start

    return total_time, iteration_times


def check_modules_equivalence(
    module_a: nn.Module,
    module_b: nn.Module,
    input_tensors: list[torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Check that `module_a` and `module_b` produce matching outputs.

    Each tensor in `input_tensors` is fed independently to both modules,
    and their outputs are compared with `torch.allclose`. Returns True if
    all outputs match within tolerance, False otherwise.
    """
    module_a.eval()
    module_b.eval()

    with torch.no_grad():
        for input_tensor in input_tensors:
            output_a = module_a(input_tensor)
            output_b = module_b(input_tensor)

            if not torch.allclose(output_a, output_b, rtol=rtol, atol=atol):
                return False

    return True
