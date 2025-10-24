import torch
import torch.distributed as dist
from typing import Optional
import os


def is_distributed():
    """Check if we're running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Get rank in distributed training, 0 for single GPU."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    """Get world size in distributed training, 1 for single GPU."""
    return dist.get_world_size() if is_distributed() else 1


def setup_distributed():
    """
    Setup distributed training if environment variables are set.
    Returns (rank, local_rank, world_size) or (0, 0, 1) for single GPU.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Running with torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, local_rank, world_size
    else:
        # Single GPU mode
        return 0, 0, 1


def cleanup_distributed():
    """Cleanup distributed training if initialized."""
    if is_distributed():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """
    Check if this is the main process (rank 0 in DDP, or single GPU).
    Useful for logging, saving checkpoints, etc.
    """
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def print_rank0(message: str):
    """Print only from rank 0 process."""
    if is_main_process():
        print(message)


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reduce tensor across all GPUs. Returns average.
    For single GPU, returns the tensor unchanged.
    """
    if not is_distributed():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.AVG)
    return rt


def barrier():
    """
    Helper function to synchronize across all processes.
    """
    if not dist.is_available() or not dist.is_initialized():
        return

    world_size = get_world_size()
    if world_size == 1:
        return

    dist.barrier()


def setup_device(local_rank: Optional[int] = None) -> torch.device:
    """
    Setup and return the appropriate device (CUDA or CPU).

    Args:
        local_rank: Local rank for DDP (None for single GPU)

    Returns:
        torch.device
    """
    if local_rank is not None:
        # DDP mode: use specific GPU
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA not available but local_rank={local_rank} specified"
            )

        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

        # Only print from rank 0 in DDP
        if local_rank == 0:
            print(f"Using device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Running in DDP mode with {dist.get_world_size()} GPUs")
    else:
        # Single GPU or CPU mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")

    return device
