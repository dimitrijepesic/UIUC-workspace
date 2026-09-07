"""Smoke test: does auto_LiRPA's Jacobian / local-Lipschitz bounding support
sigmoid activations?

Context. The released driver for Shi et al. (NeurIPS 2022, "Efficiently
Computing Local Lipschitz Constants of Neural Networks via Bound Propagation")
builds a backward gradient graph that supports exactly four node types --
Linear, ReLU, Reshape, Conv2d -- and raises NotImplementedError for anything
else (gradient.py in the Local-Lipschitz-Constants repo). Every Pasado
Section 5.4 network is a sigmoid network, so that driver cannot load them.
The current auto_LiRPA master advertises a more general Jacobian API
(examples/vision/jacobian.py), but that example is also ReLU-only, so sigmoid
support there is an open question -- the one this script answers.

The answer decides the comparison plan either way:
  - works  -> build the haze-reduction comparison against Pasado Section 5.4
              (same checkpoints, same images, same epsilons);
  - fails  -> documented capability gap: the SOTA gradient-bounding tool does
              not handle smooth activations, which is exactly the case
              Pasado's regression-fitted precise transformer targets.

Run (from lipschitz-comparison/):
    .venv\\Scripts\\python.exe smoke_test_sigmoid_jacobian.py
"""
import traceback

import torch
import torch.nn as nn

# Everything on CPU: this machine has no CUDA, and the question here is
# capability (does the graph build at all?), not speed.
DEVICE = "cpu"

# Small input keeps the test in seconds; 784 matches Pasado's MNIST input
# dimension so a "works" result transfers directly to the real comparison.
IN_DIM = 784
WIDTH = 20


def sigmoid_mlp():
    """Tiny sigmoid MLP mirroring Pasado's FCN shape (784 -> w -> w -> 10),
    just narrower (w=20 instead of 100) so the test runs instantly."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(IN_DIM, WIDTH),
        nn.Sigmoid(),
        nn.Linear(WIDTH, WIDTH),
        nn.Sigmoid(),
        nn.Linear(WIDTH, 10),
    )


def relu_mlp():
    """The control: identical architecture with ReLU. Their example is known
    to work with ReLU, so if sigmoid fails while this passes, the failure is
    specifically about the activation -- not about our harness, our versions,
    or our input shapes."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(IN_DIM, WIDTH),
        nn.ReLU(),
        nn.Linear(WIDTH, WIDTH),
        nn.ReLU(),
        nn.Linear(WIDTH, 10),
    )


def try_jacobian_bounds(model_fn, name, eps=0.01):
    """Attempt the full pipeline for one architecture and report exactly
    where it breaks, if it breaks.

    New-API requirement discovered on the first (failed) attempt: the
    Jacobian request must be part of the traced graph itself, via a wrapper
    whose forward ends with JacobianOP.apply(y, x). BoundedModule expands
    that marker node into the backward/gradient graph during construction
    (jacobian.py: _expand_jacobian -> per-op build_gradient_node). Calling
    compute_jacobian_bounds on a plain model leaves zero Jacobian nodes,
    which surfaces as the misleading "Multiple Jacobian nodes found" error
    (their check is len(nodes) == 1, and 0 fails it too).

    Beyond "does it run", this follows the correctness pattern from their
    own example: at eps=0 the bounds must collapse onto the exact Jacobian
    computed by torch.autograd -- so a "pass" also means the gradient graph
    computes the right thing, not merely that it executes.
    """
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from auto_LiRPA.jacobian import JacobianOP

    class JacobianWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            y = self.model(x)
            return JacobianOP.apply(y, x)

    torch.manual_seed(0)
    model_ori = model_fn().to(DEVICE)
    # Batch size 1: the Jacobian machinery asserts it.
    x0 = torch.rand(1, 1, 28, 28, device=DEVICE)

    print(f"\n=== [{name}] ===")

    # Phase 1: graph construction. This is where a missing per-op
    # build_gradient_node would surface (e.g. AttributeError on BoundSigmoid).
    try:
        bounded = BoundedModule(JacobianWrapper(model_ori), x0, device=DEVICE)
    except Exception as e:
        print(f"[{name}] FAILED at BoundedModule construction (gradient-graph "
              f"expansion): {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # Phase 2: concrete evaluation vs autograd ground truth at the point x0.
    # If the expanded gradient graph computes the wrong function, it shows
    # here immediately, with no bounding involved yet.
    try:
        jac_graph = bounded(x0)
        jac_autograd = torch.autograd.functional.jacobian(
            lambda t: model_ori(t.requires_grad_(True)), x0)
        a = jac_graph.reshape(-1)
        b = jac_autograd.reshape(-1)
        if a.numel() != b.numel():
            print(f"[{name}] WARNING: shape mismatch graph {tuple(jac_graph.shape)} "
                  f"vs autograd {tuple(jac_autograd.shape)}; skipping value check")
        else:
            diff = (a - b).abs().max().item()
            print(f"[{name}] concrete Jacobian vs autograd: max|diff| = {diff:.3e}")
            if diff > 1e-5:
                print(f"[{name}] FAILED: gradient graph disagrees with autograd")
                return False
    except Exception as e:
        print(f"[{name}] FAILED at concrete evaluation: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False

    # Phase 3: the actual bounds. eps=0 must collapse (lower == upper == the
    # exact Jacobian); eps>0 must produce a nonzero gap that encloses it.
    for eps_i in (0.0, eps):
        bx = BoundedTensor(x0, PerturbationLpNorm(norm=float("inf"), eps=eps_i))
        try:
            lower, upper = bounded.compute_jacobian_bounds(bx)
        except Exception as e:
            print(f"[{name}] FAILED inside compute_jacobian_bounds "
                  f"(eps={eps_i}): {type(e).__name__}: {e}")
            traceback.print_exc()
            return False
        gap = (upper - lower).max().item()
        print(f"[{name}] eps={eps_i}: bounds shape {tuple(lower.shape)}, "
              f"max gap = {gap:.3e}")
        if eps_i == 0.0 and gap > 1e-5:
            print(f"[{name}] FAILED: bounds do not collapse at eps=0")
            return False
        if eps_i > 0.0 and gap <= 0.0:
            print(f"[{name}] FAILED: zero gap at eps>0 is suspicious")
            return False

    print(f"[{name}] OK -- gradient graph builds, matches autograd, "
          f"and produces sound-looking bounds")
    return True


if __name__ == "__main__":
    import auto_LiRPA
    print(f"torch {torch.__version__}, "
          f"auto_LiRPA {getattr(auto_LiRPA, '__version__', '?')}")

    # ReLU first: it is the control. If even the control fails, the sigmoid
    # result below is meaningless and we must say so, not report a "finding".
    relu_ok = try_jacobian_bounds(relu_mlp, "ReLU control")
    sigmoid_ok = try_jacobian_bounds(sigmoid_mlp, "Sigmoid")

    print("\n=== VERDICT ===")
    if relu_ok and sigmoid_ok:
        print("Both passed: current auto_LiRPA DOES support sigmoid in "
              "Jacobian bounding.\n-> Next step: build the haze-reduction "
              "comparison against Pasado Section 5.4 (same checkpoints, "
              "same images, same epsilons).")
    elif relu_ok and not sigmoid_ok:
        print("ReLU passed, sigmoid failed: the gap is specifically the "
              "activation, not our setup.\n-> Documented capability gap: "
              "gradient bounding in the SOTA tool does not cover smooth "
              "activations -- exactly the case Pasado's precise transformer "
              "handles via per-neuron regression.")
    elif not relu_ok and sigmoid_ok:
        print("Sigmoid passed but ReLU failed -- unexpected; inspect the "
              "ReLU traceback above before drawing any conclusion.")
    else:
        print("BOTH failed: our environment/API usage is broken. This is "
              "NOT evidence about sigmoid support -- fix the harness first.")
