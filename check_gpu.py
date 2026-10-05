"""Check CUDA visibility and execute a small operation on every visible GPU."""

def main():
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is missing. Activate your person_detection environment.")
    print(f"PyTorch: {torch.__version__} | Built for CUDA: {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA unavailable to this Python environment. Check nvidia-smi and your PyTorch installation.")
    for device in range(torch.cuda.device_count()):
        try:
            x = torch.ones((32, 32), device=f"cuda:{device}")
            result = x @ x
            torch.cuda.synchronize(device)
            assert result[0, 0].item() == 32
            free, total = torch.cuda.mem_get_info(device)
            print(f"PASS GPU {device}: {torch.cuda.get_device_name(device)} | "
                  f"VRAM free/total: {free / 2**20:.0f}/{total / 2**20:.0f} MiB | CUDA computation OK")
            del x, result
        except Exception as error:
            raise SystemExit(f"FAIL GPU {device}: {error}")


if __name__ == "__main__":
    main()
