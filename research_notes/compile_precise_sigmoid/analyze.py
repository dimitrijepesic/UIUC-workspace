"""Local analysis of torch.compile on batched_splits.precise_sigmoid (Pasado a57b560).

Eager A2 inside (USE_CLOSED_FORM_FIT=True, USE_COMPILED_FIT=False, USE_AUTO_PLANE_FIT=False),
explicit noise tensor. No production code is changed.
"""
import json, os, sys, time
import numpy as np
import torch
import torch._dynamo
import torch._inductor.metrics as imetrics

REPO = "/Users/dimitrije/prakse/UIUC/Pasado"
for p in ("forward_mode_tensorized_src", "experiments", "Section_5_4"):
    sys.path.insert(0, os.path.join(REPO, p))
import batched_splits as bs
import compare_lstsq_solvers as h
from measure_split_sweep import load_network_and_images

h.set_production_flags()
torch.set_default_dtype(torch.float64)
bs.USE_CLOSED_FORM_FIT, bs.USE_COMPILED_FIT, bs.USE_AUTO_PLANE_FIT = True, False, False
for name in ("cache_size_limit", "recompile_limit"):
    if hasattr(torch._dynamo.config, name):
        setattr(torch._dynamo.config, name, 64)

report = dict(commit=os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
              torch=torch.__version__, device="cpu", cases=[])


def np_state():
    s = np.random.get_state()
    return s[0], s[1].copy(), s[2], s[3], s[4]


def same_np(a, b):
    return a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def inputs(network):
    """(label, z, noise) for the first and the second sigmoid layer, S = 1."""
    net, images = load_network_and_images(network, torch.device("cpu"), torch.float64, 1)
    layers = bs.extract_affine_layers(net, network)
    img = images[0][1]
    lows, highs = torch.tensor([0.0]), torch.tensor([h.EPS_SCHEDULE[0]])
    out = []
    np.random.seed(7)
    z = bs.hazed_input(img, lows, highs)
    for k, (w, b) in enumerate(layers[:2]):
        zin = bs.affine(z, w, b)
        noise = bs._perturbation(1, w.shape[1], 7, torch.float64, torch.device("cpu"))
        out.append((f"{network} layer {k + 1}: {zin.rg.shape[2]} boxes, G0 {zin.rg.shape[1]}", zin, noise))
        z, _ = bs.precise_sigmoid(zin, noise)
    return layers, img, out


def max_rel(a, b):
    d = (a - b).abs().max().item()
    s = b.abs().max().item()
    return d / s if s else d


compiled = torch.compile(bs.precise_sigmoid, fullgraph=True)
for network in ("3layer", "big"):
    layers, img, cases = inputs(network)
    for label, z, noise in cases:
        case = dict(label=label)
        torch._dynamo.reset()
        ex = torch._dynamo.explain(bs.precise_sigmoid)(z, noise)
        case["explain"] = dict(graphs=ex.graph_count, breaks=ex.graph_break_count,
                               ops_per_graph=[len(o) for o in ex.ops_per_graph],
                               break_reasons=[r.reason.splitlines()[0][:200] for r in ex.break_reasons])
        torch._dynamo.reset()
        imetrics.reset()
        before = np_state()
        try:
            t0 = time.perf_counter()
            out_c, ok_c = compiled(z, noise)
            case["compile_s"] = time.perf_counter() - t0
            case["fullgraph"] = "ok"
        except Exception as e:
            case["fullgraph"] = f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"
            report["cases"].append(case)
            print(json.dumps(case, indent=1))
            continue
        case["inductor_kernels"] = imetrics.generated_kernel_count
        case["numpy_state_untouched"] = same_np(before, np_state())
        out_e, ok_e = bs.precise_sigmoid(z, noise)
        case["nonzero_flag"] = dict(eager=bool(ok_e), compiled=bool(ok_c))
        case["output_rel_diff"] = {k: max_rel(getattr(out_c, k), getattr(out_e, k)) for k in ("rc", "rg", "dc", "dg")}
        case["output_bitwise"] = {k: bool(torch.equal(getattr(out_c, k), getattr(out_e, k))) for k in ("rc", "rg", "dc", "dg")}
        case["output_shapes_equal"] = all(getattr(out_c, k).shape == getattr(out_e, k).shape for k in ("rc", "rg", "dc", "dg"))
        report["cases"].append(case)
        print(json.dumps(case, indent=1), flush=True)

    # final bounds: whole batched pass with the compiled precise_sigmoid swapped in
    eps = h.EPS_SCHEDULE[0]
    res = {}
    for S in (1, 4):
        np.random.seed(0)
        eager_b = bs.split_bounds(layers, img, eps, S, seed=7)
        st_e = np_state()
        orig = bs.precise_sigmoid
        bs.precise_sigmoid = compiled
        try:
            np.random.seed(0)
            comp_b = bs.split_bounds(layers, img, eps, S, seed=7)
            st_c = np_state()
        finally:
            bs.precise_sigmoid = orig
        res[f"S={S}"] = dict(max_rel_diff=max_rel(comp_b, eager_b), bitwise=bool(torch.equal(comp_b, eager_b)),
                             numpy_state_same_as_eager=same_np(st_e, st_c))
    report[f"{network}_final_bounds"] = res
    print(network, "final bounds", res, flush=True)

json.dump(report, open(os.path.join(os.path.dirname(__file__), "analysis_cpu.json"), "w"), indent=1, default=str)
