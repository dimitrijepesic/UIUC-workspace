"""Local check (CPU) of the v2 fix for G4: the generator dimension G0 marked dynamic before
every compiled call of batched_splits.precise_sigmoid (Pasado a57b560, nothing changed).

Per network and S: one graph for every sigmoid layer; first, second and third pass
bit-identical; fullgraph; outputs against eager; and the time of the dynamic graph against
a graph with static shapes compiled for each G0, on the inputs of every layer.
Writes check_mark_dynamic.json next to this file.
"""
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import torch._dynamo

REPO = "/Users/dimitrije/prakse/UIUC/Pasado"
for p in ("forward_mode_tensorized_src", "experiments", "Section_5_4"):
    sys.path.insert(0, os.path.join(REPO, p))
import batched_splits as bs  # noqa: E402
import precise_transformer as pt  # noqa: E402
from measure_split_sweep import load_network_and_images  # noqa: E402
from test_batched_splits import EPS_SCHEDULE  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

for flag in ("USE_VECTORIZED_PRECISE", "USE_VECTORIZED_BOUNDARY", "USE_BATCHED_LSTSQ", "USE_BATCHED_GRID",
             "USE_REAL_CUBIC"):
    setattr(pt, flag, True)
torch.set_default_dtype(torch.float64)
bs.USE_CLOSED_FORM_FIT, bs.USE_COMPILED_FIT, bs.USE_AUTO_PLANE_FIT = True, False, False
EAGER = bs.precise_sigmoid
OUTPUTS = ("rc", "rg", "dc", "dg")
REPEATS = 15


def marked(fn):
    def run(z, noise):
        torch._dynamo.mark_dynamic(z.rg, 1)
        torch._dynamo.mark_dynamic(z.dg, 1)
        return fn(z, noise)
    return run


def graphs():
    return torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)


def normwise(a, b):
    s = b.abs().max().item()
    d = (a - b).abs().max().item()
    return d / s if s else d


def median_ms(fn, *args):
    fn(*args)
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args)
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def capture_inputs(layers, img, eps, S):
    """(z, noise) of every sigmoid layer of an eager pass."""
    seen = []

    def rec(z, noise):
        seen.append((z, noise))
        return EAGER(z, noise)

    bs.precise_sigmoid = rec
    try:
        bs.split_bounds(layers, img, eps, S, seed=7)
    finally:
        bs.precise_sigmoid = EAGER
    return seen


report = dict(torch=torch.__version__, device="cpu", repeats=REPEATS, cases=[])
for network in ("3layer", "4layer", "5layer", "big"):
    net, images = load_network_and_images(network, torch.device("cpu"), torch.float64, 1)
    layers = bs.extract_affine_layers(net, network)
    img = images[0][1]
    eps = EPS_SCHEDULE[8]
    for S in (1, 8):
        case = dict(network=network, S=S)
        torch._dynamo.reset()
        g0 = graphs()
        comp = marked(torch.compile(EAGER, fullgraph=True))
        outs = {}

        def recorder(tag):
            def fn(z, noise):
                out, ok = comp(z, noise)
                outs.setdefault(tag, []).append(out)
                return out, ok
            return fn

        regions, bounds = {}, {}
        for tag in ("pass1", "pass2", "pass3"):
            bs.precise_sigmoid = recorder(tag)
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                bounds[tag] = bs.split_bounds(layers, img, eps, S, seed=7)
            bs.precise_sigmoid = EAGER
            regions[tag] = [e.name.split(": ")[1] for e in sorted(prof.events(), key=lambda e: e.time_range.start)
                            if e.name.startswith("Torch-Compiled Region")]
        case["graphs"] = graphs() - g0
        case["regions"] = regions
        case["one_graph_every_layer"] = len({r for rs in regions.values() for r in rs}) == 1
        case["bounds_bits_equal"] = bool(torch.equal(bounds["pass1"], bounds["pass2"])
                                         and torch.equal(bounds["pass2"], bounds["pass3"]))
        case["outputs_bits_equal"] = all(torch.equal(getattr(a, k), getattr(b, k))
                                         for t in ("pass2", "pass3")
                                         for a, b in zip(outs["pass1"], outs[t]) for k in OUTPUTS)
        inputs = capture_inputs(layers, img, eps, S)
        layer_rows = []
        for k, (z, noise) in enumerate(inputs):
            out_e, _ = EAGER(z, noise)
            out_c, _ = comp(z, noise)
            static = torch.compile(EAGER, fullgraph=True, dynamic=False)
            row = dict(layer=k + 1, G0=z.rg.shape[1],
                       rel_vs_eager={n: normwise(getattr(out_c, n), getattr(out_e, n)) for n in OUTPUTS},
                       eager_ms=median_ms(EAGER, z, noise),
                       static_ms=median_ms(static, z, noise),
                       dynamic_ms=median_ms(comp, z, noise))
            row["dynamic_over_static"] = row["dynamic_ms"] / row["static_ms"]
            layer_rows.append(row)
        case["layers"] = layer_rows
        report["cases"].append(case)
        print(json.dumps({k: v for k, v in case.items() if k != "regions"}, default=str), flush=True)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_mark_dynamic.json"), "w") as f:
    json.dump(report, f, indent=1, default=str)
