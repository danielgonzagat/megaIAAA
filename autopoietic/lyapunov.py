def V(x: float) -> float:
    """Simple Lyapunov function.

    Uses V(x) = 0.5 * x^2 which is positive definite.
    """
    return 0.5 * x ** 2


def vdot(x: float, dx_dt: float) -> float:
    """Time derivative of ``V``.

    For V(x) = 0.5*x^2, the derivative is x * dx/dt.
    """
    return x * dx_dt
