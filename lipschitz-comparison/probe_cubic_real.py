"""Compare the complex64 Cardano root solve against a real trigonometric one.

inverse_poly_tensor solves the depressed cubic u^3 - u/4 - y/2 = 0 (that is,
sigma''(x) = y after substituting s = sigma(x) = u + 1/2) using Cardano's
formula evaluated in complex64, then discards whichever roots did not come out
real. Two consequences:

  * the roots are effectively computed in SINGLE precision inside an analysis
    that is float64 everywhere else;
  * complex arithmetic is a hard blocker for TorchInductor codegen.

For this particular cubic, p = -1/4 and q = -y/2, so all three roots are real
exactly when 4p^3 + 27q^2 < 0, i.e. |y| < 1/(6*sqrt(3)) ~ 0.09623 -- which is
precisely the maximum of |sigma''|. So over the whole physically reachable
range the complex path is computing three real numbers the long way round, and
Viete's trigonometric form gives them directly:

    u_k = (1/sqrt(3)) * cos( acos(6*sqrt(3)*y)/3 - 2*pi*k/3 ),  k = 0,1,2

with the acos argument landing in [-1, 1] exactly when the roots are real.

This script does not assume the replacement is correct -- the existing
behaviour is the specification. It measures, over in-range and out-of-range y:
  1. do both agree on WHICH roots are real (the NaN pattern downstream)?
  2. where both give a root, how far apart are they?
  3. which one actually satisfies the cubic more accurately (residual test)?
"""
import math

import torch

torch.set_default_dtype(torch.float64)


def cuberoot(x):
    return x ** 0.33333          # verbatim from precise_transformer


def inverse_poly_complex(y):
    """Verbatim port of the current inverse_poly_tensor."""
    Q = torch.tensor(-1 / 12.)
    R = 0.25 * y
    R = R.type(torch.complex64)
    J = torch.tensor(1.j)
    sqrt_arg = ((Q ** 3) + (R ** 2))
    sqrt_arg = sqrt_arg.type(torch.complex64)
    S = cuberoot(R + torch.sqrt(sqrt_arg))
    T = cuberoot(R - torch.sqrt(sqrt_arg))
    x1 = S + T + 0.5
    sqrt_3_by_2 = torch.sqrt(torch.tensor(3.)) / 2.
    x2 = -0.5 * (S + T) + 0.5 + (sqrt_3_by_2 * J) * (S - T)
    x3 = -0.5 * (S + T) + 0.5 - (sqrt_3_by_2 * J) * (S - T)
    return (x1, x2, x3)


def inverse_poly_real(y):
    """Viete's trigonometric form, real arithmetic in y's own dtype.

    Returns the three roots as real tensors, NaN where the root does not exist
    (|arg| > 1, i.e. y outside the range of sigma'' -- only one real root then,
    and the analysis discards the other two anyway).
    """
    arg = 6.0 * math.sqrt(3.0) * y
    # Outside [-1, 1] there is a single real root; acos would produce NaN, which
    # is exactly the value the downstream filter substitutes for a complex root.
    phi = torch.acos(arg.clamp(-1.0, 1.0)) / 3.0
    phi = torch.where(arg.abs() <= 1.0, phi, torch.full_like(phi, float("nan")))
    scale = 1.0 / math.sqrt(3.0)
    two_pi_3 = 2.0 * math.pi / 3.0
    return tuple(scale * torch.cos(phi - k * two_pi_3) + 0.5 for k in (0, 1, 2))


def residual(s, y):
    """2s^3 - 3s^2 + s - y; zero iff s solves sigma''=y in s-space."""
    return 2.0 * s ** 3 - 3.0 * s ** 2 + s - y


def report(name, ys):
    print(f"\n=== {name}  (n={ys.numel()}) ===")
    cx = inverse_poly_complex(ys)
    rl = inverse_poly_real(ys)

    # Which roots does each path consider real? The production filter is
    # torch.isreal on the complex tensor; for the real path it is "not NaN".
    for k in range(3):
        cx_real = torch.isreal(cx[k])
        rl_real = ~torch.isnan(rl[k])
        agree = (cx_real == rl_real).float().mean().item() * 100
        both = cx_real & rl_real
        if both.any():
            # Roots are unordered between the two formulas, so compare each
            # complex-path root against the closest real-path root.
            stack = torch.stack([rl[j][both] for j in range(3)])
            d = (stack - cx[k].real[both]).abs().min(dim=0).values
            dmax, dmed = d.max().item(), d.median().item()
        else:
            dmax = dmed = float("nan")
        print(f"  root {k}: real-flag agreement {agree:5.1f}%   "
              f"both-real {int(both.sum()):5d}   "
              f"closest-match median {dmed:.2e} max {dmax:.2e}")

    # Accuracy: how well does each satisfy the cubic it is supposed to solve?
    cx_res, rl_res = [], []
    for k in range(3):
        m = torch.isreal(cx[k])
        if m.any():
            cx_res.append(residual(cx[k].real[m].double(), ys[m]).abs())
        m2 = ~torch.isnan(rl[k])
        if m2.any():
            rl_res.append(residual(rl[k][m2], ys[m2]).abs())
    cx_res = torch.cat(cx_res) if cx_res else torch.tensor([float("nan")])
    rl_res = torch.cat(rl_res) if rl_res else torch.tensor([float("nan")])
    print(f"  |residual| complex64 path : median {cx_res.median():.3e}  "
          f"max {cx_res.max():.3e}")
    print(f"  |residual| real trig path : median {rl_res.median():.3e}  "
          f"max {rl_res.max():.3e}")


if __name__ == "__main__":
    g = torch.Generator().manual_seed(3)
    LIM = 1.0 / (6.0 * math.sqrt(3.0))
    report("in range |y| < 1/(6*sqrt3): three real roots",
           (torch.rand(20000, generator=g) * 2 - 1) * LIM * 0.999)
    report("out of range |y| > limit: one real root",
           torch.sign(torch.randn(20000, generator=g))
           * (LIM * 1.001 + torch.rand(20000, generator=g) * 5))
    report("near the boundary |y| ~ limit",
           torch.sign(torch.randn(20000, generator=g))
           * LIM * (1 + (torch.rand(20000, generator=g) - 0.5) * 1e-6))
    report("y == 0 exactly", torch.zeros(1000))
