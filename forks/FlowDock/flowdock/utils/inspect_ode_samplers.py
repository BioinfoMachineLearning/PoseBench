import argparse

import torch
from beartype import beartype
from beartype.typing import Literal


def clamp_tensor(value: torch.Tensor, min: float = 1e-6, max: float = 1 - 1e-6) -> torch.Tensor:
    """Set the upper and lower bounds of a tensor via clamping.

    :param value: The tensor to clamp.
    :param min: The minimum value to clamp to. Default is `1e-6`.
    :param max: The maximum value to clamp to. Default is `1 - 1e-6`.
    :return: The clamped tensor.
    """
    return value.clamp(min=min, max=max)


@beartype
def main(
    start_time: float, num_steps: int, sampler: Literal["ODE", "VDODE"] = "VDODE", eta: float = 1.0
):
    """Inspect different ODE samplers by printing the left hand side (LHS) and right hand side.

    (RHS) of their time ratio schedules. Note that the LHS and RHS are clamped to the range
    `[1e-6, 1 - 1e-6]` by default.

    :param start_time: The start time of the ODE sampler.
    :param num_steps: The number of steps to take.
    :param sampler: The ODE sampler to use.
    :param eta: The variance diminishing factor.
    """
    assert 0 < start_time <= 1.0, "The argument `start_time` must be in the range (0, 1]."
    schedule = torch.linspace(start_time, 0, num_steps + 1)
    for t, s in zip(schedule[:-1], schedule[1:]):
        if sampler == "ODE":
            # Baseline ODE
            print(
                f"t: {t:.3f}; s: {s:.3f}; LHS -> (1 - t) * x0_hat: {clamp_tensor((1 - t)):.3f}; RHS -> t * xt: {clamp_tensor(t):.3f}"
            )
        elif sampler == "VDODE":
            # Variance Diminishing (VD) ODE
            print(
                f"t: {t:.3f}; s: {s:.3f}; LHS -> (1 - ((s / t) * eta)) * x0_hat: {clamp_tensor(1 - ((s / t) * eta)):.3f}; RHS -> ((s / t) * eta) * xt: {clamp_tensor((s / t) * eta):.3f}"
            )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--start_time", type=float, default=1.0)
    argparser.add_argument("--num_steps", type=int, default=20)
    argparser.add_argument("--sampler", type=str, choices=["ODE", "VDODE"], default="VDODE")
    argparser.add_argument("--eta", type=float, default=1.0)
    args = argparser.parse_args()
    main(args.start_time, args.num_steps, sampler=args.sampler, eta=args.eta)
