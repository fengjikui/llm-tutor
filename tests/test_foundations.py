from llm_tutor.foundations.manual_gradient_descent import (
    train_linear_regression_by_hand,
    train_linear_regression_with_autograd,
)


def test_manual_gradient_descent_matches_autograd() -> None:
    manual = train_linear_regression_by_hand(steps=80)[-1]
    autograd = train_linear_regression_with_autograd(steps=80)[-1]

    assert abs(manual.weight - autograd.weight) < 1e-6
    assert abs(manual.bias - autograd.bias) < 1e-6
    assert abs(manual.loss - autograd.loss) < 1e-6
    assert abs(manual.weight - 3.0) < 0.1
    assert abs(manual.bias + 2.0) < 0.1
