from __future__ import annotations

from llm_tutor.foundations.manual_gradient_descent import (
    train_linear_regression_by_hand,
    train_linear_regression_with_autograd,
)


def print_history(title: str, history) -> None:
    print(f"\n{title}")
    for row in history:
        print(
            f"step={row.step:03d} "
            f"w={row.weight:+.3f} "
            f"b={row.bias:+.3f} "
            f"loss={row.loss:.4f}"
        )


def main() -> None:
    print_history("manual gradient descent", train_linear_regression_by_hand())
    print_history("PyTorch autograd", train_linear_regression_with_autograd())


if __name__ == "__main__":
    main()
