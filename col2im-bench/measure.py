import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
BUILD = Path(os.environ.get("OPENCV_BUILD_DIR", ROOT.parent.parent.parent / ".tools/opencv-col2im-build"))
BEFORE = Path(os.environ.get("OPENCV_BEFORE_DIR", ROOT / "baseline"))
PROBE = Path(os.environ.get("OPENCV_PROBE", ROOT / "probe"))


def run(case, threads, iterations, variant, cases="bench_cases.txt", wide=False, net=False):
    env = dict(os.environ, LD_LIBRARY_PATH=str(BUILD / "lib"), OPENCV_LOG_LEVEL="ERROR")
    for key in ("PROBE_WIDE_VALUES", "PROBE_NET"):
        env.pop(key, None)
    if variant == "before":
        env["LD_LIBRARY_PATH"] = str(BEFORE) + ":" + env["LD_LIBRARY_PATH"]
    if wide:
        env["PROBE_WIDE_VALUES"] = "1"
    if net:
        env["PROBE_NET"] = "1"
    with tempfile.TemporaryDirectory(prefix="opencv-deconv-") as temp:
        output = Path(temp) / "output.bin"
        process = subprocess.run([
            "taskset", "-c", "0" if threads == 1 else "0-3", str(PROBE),
            str(ROOT / cases), case, str(threads), str(iterations), str(output)
        ], env=env, text=True, capture_output=True)
        if process.returncode:
            raise RuntimeError(f"{variant}/{case}/{threads}: {process.stdout}\n{process.stderr}")
        rows = [json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")]
        phases = [json.loads(line[len("DECONV_PHASE "):])
                  for line in process.stderr.splitlines() if line.startswith("DECONV_PHASE ")]
        if phases:
            assert len(rows) == 1
            rows[0]["phase_samples"] = phases
        data = output.read_bytes()
        for row in rows:
            row.update(variant=variant, wide=wide)
        return rows, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "bench"))
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--cases", default="pointwise_tiny,pointwise,ocr128,ocr368,overlap,dilated,three_d,tiny")
    parser.add_argument("--case-file", default="bench_cases.txt")
    parser.add_argument("--threads", default="1,4")
    parser.add_argument("--net", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    rows, summary = [], []
    output = ROOT / (args.output or (args.mode + ".json"))
    if args.mode == "verify":
        for wide in (False, True):
            for threads in map(int, args.threads.split(",")):
                before, bdata = run("all", threads, 2, "before", "verify_cases.txt", wide)
                after, adata = run("all", threads, 2, "after", "verify_cases.txt", wide)
                assert len(before) == len(after) == 192
                assert bdata == adata, (wide, threads)
                assert [x["case"] for x in before] == [x["case"] for x in after]
                item = dict(wide=wide, threads=threads, cases=len(before),
                            compared_values=len(bdata) // 4, bitwise_equal=True,
                            output_sha256=hashlib.sha256(bdata).hexdigest())
                rows.extend(before + after)
                summary.append(item)
                print(json.dumps(item), flush=True)
    else:
        for threads in map(int, args.threads.split(",")):
            for case in args.cases.split(","):
                pairs = []
                for repeat in range(args.pairs):
                    pair, data = {}, {}
                    for variant in (("before", "after") if repeat % 2 == 0 else ("after", "before")):
                        values, result = run(case, threads, args.iterations, variant,
                                             cases=args.case_file, net=args.net)
                        assert len(values) == 1
                        pair[variant] = values[0]
                        data[variant] = result
                        values[0]["repeat"] = repeat
                        rows.extend(values)
                    assert data["before"] == data["after"], (case, threads)
                    pairs.append(pair)
                reductions = [(1 - p["after"]["median_ms"] / p["before"]["median_ms"]) * 100 for p in pairs]
                item = dict(case=case, threads=threads,
                            before_ms=statistics.median(p["before"]["median_ms"] for p in pairs),
                            after_ms=statistics.median(p["after"]["median_ms"] for p in pairs),
                            median_reduction_pct=statistics.median(reductions),
                            min_reduction_pct=min(reductions), max_reduction_pct=max(reductions))
                summary.append(item)
                print(json.dumps(item), flush=True)
                output.write_text(json.dumps(dict(args=vars(args), rows=rows, summary=summary), indent=2) + "\n")
    output.write_text(json.dumps(dict(args=vars(args), rows=rows, summary=summary), indent=2) + "\n")


if __name__ == "__main__":
    main()
