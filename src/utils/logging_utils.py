import inspect
import logging
import os


def _process_index() -> int:
    """Return torch.distributed rank, falling back to env vars or 0."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return int(os.environ.get("RANK", "0"))


def log_for_0(msg, *args, level=logging.INFO):
    """Log only on the first process (rank == 0)."""
    if _process_index() != 0:
        return
    caller_module = inspect.currentframe().f_back.f_globals.get("__name__", __name__)
    logging.getLogger(caller_module).log(level, msg, *args)


def add_file_logging(output_dir: str):
    """Mirror rank-0 logs to output_dir/log.txt."""
    if _process_index() != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(output_dir, "log.txt"), mode="a")
    handler.setFormatter(logging.Formatter("%(levelname)s - %(name)s - %(message)s"))
    logging.getLogger().addHandler(handler)
