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
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
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


def barrier():
    """Synchronize all processes (no-op for single GPU)."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """
    Reduce dictionary values across all processes.
    Useful for gathering metrics from all GPUs.

    Args:
        input_dict: Dictionary with tensor or float values
        average: If True, compute average; if False, sum

    Returns:
        Dictionary with reduced values (only valid on rank 0)
    """
    if not dist.is_available() or not dist.is_initialized():
        return input_dict

    world_size = get_world_size()
    if world_size == 1:
        return input_dict

    # Convert to tensors
    names = []
    values = []
    for k, v in sorted(input_dict.items()):
        names.append(k)
        if isinstance(v, torch.Tensor):
            values.append(v.detach().clone())
        else:
            values.append(torch.tensor(v))

    # Stack and reduce
    values = torch.stack(values, dim=0)
    if values.device.type == "cpu":
        # Move to GPU for NCCL
        device = torch.device(f"cuda:{get_rank()}")
        values = values.to(device)

    dist.all_reduce(values)

    if average:
        values /= world_size

    # Convert back to dict
    reduced_dict = {k: v.item() for k, v in zip(names, values)}
    return reduced_dict


def synchronize():
    """
    Helper function to synchronize (barrier) across all processes.
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