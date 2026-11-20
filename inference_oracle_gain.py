"""Compatibility entry point for unified Q inference with oracle gain.

The implementation and shared helpers live in inference_orthogonal_power.py.
"""

from inference_orthogonal_power import (
    image_metrics, load_input, oracle_receive, save_image, save_power_allocation,
)


def run(args):
    from inference_orthogonal_power import run_inference
    if getattr(args, "receiver_mode", None) is None:
        args.receiver_mode = "oracle"
    return run_inference(args)


def main():
    from inference_orthogonal_power import main as unified_main
    return unified_main(default_receiver="oracle")


if __name__ == "__main__":
    main()
