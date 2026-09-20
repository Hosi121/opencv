from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent


def row(name, spatial, kernel, stride=None, dilation=None, pads=None, adjust=None,
        batch=1, ci=4, co=6, groups=2, bias=True, dynamic=False):
    dims = len(spatial)
    stride = stride or [1] * dims
    dilation = dilation or [1] * dims
    pads = pads or [0] * (2 * dims)
    adjust = adjust or [0] * dims
    fields = [name, dims, batch, ci, co, groups, int(bias), int(dynamic)]
    for values in [spatial, kernel, stride, dilation, pads, adjust]:
        fields.extend(values)
    return " ".join(map(str, fields))


def main():
    cases = []
    rng = random.Random(20260920)
    for dims in (2, 3):
        for index in range(48):
            spatial = [rng.randint(2, 8) for _ in range(dims)]
            kernel = [rng.randint(1, 4) for _ in range(dims)]
            stride = [rng.randint(1, 3) for _ in range(dims)]
            dilation = [rng.randint(1, 2) for _ in range(dims)]
            pads = [rng.randint(0, 1) for _ in range(2 * dims)]
            adjust = [rng.randrange(s) for s in stride]
            if index < 6:
                kernel = stride = dilation = [1] * dims
                pads, adjust = [0] * (2 * dims), [0] * dims
            for dynamic in (False, True):
                cases.append(row(f"d{dims}_{index}_{int(dynamic)}", spatial, kernel, stride,
                                 dilation, pads, adjust, batch=1 + index % 2,
                                 ci=4 if index % 3 else 6, co=6 if index % 3 else 9,
                                 groups=2 if index % 3 else 3, bias=index % 2 == 0, dynamic=dynamic))
    (ROOT / "verify_cases.txt").write_text("\n".join(cases) + "\n")
    bench = [
        row("pointwise_tiny", [1, 1], [1, 1], ci=1, co=1, groups=1),
        row("pointwise", [128, 128], [1, 1], ci=16, co=16, groups=1),
        row("ocr128", [128, 128], [2, 2], [2, 2], ci=16, co=16, groups=1),
        row("ocr368", [368, 368], [2, 2], [2, 2], ci=16, co=16, groups=1),
        row("overlap", [64, 64], [3, 3], pads=[1, 1, 1, 1], ci=16, co=16, groups=1),
        row("dilated", [48, 65], [2, 3], [2, 1], [2, 2], [1, 0, 0, 1], ci=16, co=16, groups=1),
        row("three_d", [16, 16, 16], [3, 3, 3], ci=8, co=8, groups=1),
        row("tiny", [1, 1], [2, 2], [2, 2], ci=1, co=1, groups=1),
    ]
    (ROOT / "bench_cases.txt").write_text("\n".join(bench) + "\n")


if __name__ == "__main__":
    main()
