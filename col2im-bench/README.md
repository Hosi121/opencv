# CPU Deconvolution measurements

The base is OpenCV `5.x` at `8d126c4a672936fde5a21d9133c352b18d4299a5`.
The current patch is `27dadfe394f64fe4bfa7b64137d2e85e235b9f4f`.
The previous PR patch is `623c587ad91f3290e36a2c4f70163c9e30c687c2`.
The patch changes the legacy CPU `Deconvolution` layer. The new ONNX
`ConvTranspose2` layer has a separate implementation. These are layer results,
not complete-model results. ARM hardware was not tested.

The code builds valid tap tables for each axis and shares them across channels
and workers. It decodes the outer coordinates once per output row. For horizontal
stride 1 or 2, it uses 128-bit SIMD where four inputs are consecutive and the tap
set stays the same. The last change keeps the tap lists fixed across a whole
input run. It adds no buffer. Other positions use the scalar loop. Each output keeps the
original sum order. The pointwise path adds bias one channel segment at a time.
GEMM and output ownership do not change.

The test host is an Intel Core Ultra 7 255H with Ubuntu 24.04 under WSL2.
The compiler is GCC 13.3.0. The build uses Release mode, SSE3 with AVX2 dispatch,
and the pthreads backend. IPP, OpenCL, and external BLAS libraries are disabled.
The SIMD run table adds one `size_t` per output column when used. For the 128-square
2x2 case, this is 2,048 bytes shared by all workers. This is an array size,
not a process memory measurement.

## Results and limits

Current patch: one-thread layer results against upstream, from
`run_reuse_20260925/bench_base.json`:

| Input, channels in/out | Kernel, stride | Base | PR | Speedup |
| --- | --- | ---: | ---: | ---: |
| 128×128, 16/16 | 1×1, 1 | 2.972 ms | 0.095 ms | 31.4× |
| 128×128, 16/16 | 2×2, 2 | 75.136 ms | 0.879 ms | 85.5× |
| 64×64, 16/16 | 3×3, 1, pad 1 | 10.534 ms | 0.329 ms | 32.1× |
| 16×16×16, 8/8 | 3×3×3, 1 | 22.359 ms | 0.666 ms | 33.6× |

Results for the previous patch (`623c587`) are in `row_simd_20260925/`:

| File | Comparison |
| --- | --- |
| `bench_base_final.json` | Upstream `8d126c4a` to the final patch |
| `bench_final.json` | Previous PR `ae8e518f` to the final patch |
| `bench_row.json` | Previous PR to row-only commit `7a21734a` |
| `bench_simd_vs_row.json` | Row-only commit to the final patch |
| `bench_net_final.json` | Upstream to final, through a single-layer `Net` |
| `bench_edges_final.json` | Short rows, horizontal stride 1, 2, and 3 |
| `bench_tiny_final.json`, `bench_tiny_control.json` | Small-input repeat and same-binary control |
| `verify_final.json` | 768 output comparisons with upstream |
| `table_storage.json` | Table sizes for the measured shapes |

The two main comparisons use five pairs of separate processes per case, with
two warmup calls and 15 measured calls per process. Version order alternates.
Reported times are medians of process medians. Reductions are medians of paired
reductions. The files also contain each sample and the minimum and maximum
paired reductions. The row and Net comparisons use three pairs. CPU affinity
is 0 for one thread, and 0 through 3 for four threads. The host is not an isolated
benchmark server. No build ran during measurement.

The measured `forward()` call includes GEMM, table setup, output reconstruction,
and bias. Layer setup and input, weight, and output allocation are outside the
timer. The Net probe sets the input again before each measured forward call;
input setup is outside the timer. Both probes compare every output byte.

Small inputs do not show a stable gain. Some pairs are slower. The longer repeat
uses nine pairs and 1,001 calls per process. Its four-thread differences are close
to zero, but the same-binary control also varies. Do not claim no regression for
all small inputs.

The first SIMD version built a run table even when no four-input run could fit.
The large-kernel, small-input case was about 9% slower in `bench_edges.json`.
The final version skips that table for these shapes. The regression did not repeat
in `bench_edges_final.json`. The other edge cases still have small timing changes.

`profile.json` and `profile_summary.json` are diagnostic results from the SIMD
version before this guard. The source patch is `repro/profile.patch`. Timer output
is inside the forward call; do not use those total times as layer benchmarks.
The old `review_20260924/` and root result files describe earlier versions.

Results for the latest change are in `run_reuse_20260925/`. Its
`bench_confirm.json` compares the previous patch with the current commit. It uses
seven pairs and 101 calls per process. One-thread layer time is 0.957 to 0.864 ms
for the 128-square 2x2 case, and 7.768 to 7.133 ms for the 368-square case.
Four-thread times vary. The dilation case changes sign between repeats, and the
same-binary control varies by about 30 percent in each direction. Do not use its
large four-thread gains as a performance claim. Small shapes also have small
negative differences. See `bench_small_repeat.json`, `bench_control.json`, and
`bench_edges.json`.

The current commit passes all 58 accuracy tests at one and four threads, the six
performance tests, and 768 output comparisons with upstream. The test count did
not change. The existing overlap case now uses a wider row after its shape change
to cover more than one SIMD block.

## Repeat the measurements

Use Linux, GCC, CMake, Ninja, Python 3, and `taskset`. No test images or models
are required. Save the files from this directory in a local `bench` directory.
Set the three paths below to absolute paths. The source must contain the patch
and its two new test files.

```bash
export OPENCV_SOURCE_DIR=/path/to/opencv
export OPENCV_BUILD_DIR=/path/to/opencv-build
export OPENCV_BEFORE_DIR=/path/to/bench/baseline
cd /path/to/bench
mkdir -p "$OPENCV_BEFORE_DIR"

# Build the base with the same configuration and test files.
cp "$OPENCV_SOURCE_DIR/modules/dnn/src/layers/deconvolution_layer.cpp" deconvolution_after.cpp
git -C "$OPENCV_SOURCE_DIR" show \
  8d126c4a672936fde5a21d9133c352b18d4299a5:modules/dnn/src/layers/deconvolution_layer.cpp \
  > "$OPENCV_SOURCE_DIR/modules/dnn/src/layers/deconvolution_layer.cpp"
bash configure.sh
cmake --build "$OPENCV_BUILD_DIR" --target opencv_dnn opencv_test_dnn opencv_perf_dnn -j 6
cp -a "$OPENCV_BUILD_DIR"/lib/libopencv_dnn.so* "$OPENCV_BEFORE_DIR/"

# Restore the patch. Keep the other shared libraries from the same build.
cp deconvolution_after.cpp "$OPENCV_SOURCE_DIR/modules/dnn/src/layers/deconvolution_layer.cpp"
touch "$OPENCV_SOURCE_DIR/modules/dnn/src/layers/deconvolution_layer.cpp"
cmake --build "$OPENCV_BUILD_DIR" --target opencv_dnn opencv_test_dnn opencv_perf_dnn -j 6

g++ -O2 -std=c++17 probe.cpp \
  -I"$OPENCV_SOURCE_DIR/modules/core/include" \
  -I"$OPENCV_SOURCE_DIR/modules/dnn/include" -I"$OPENCV_BUILD_DIR" \
  -L"$OPENCV_BUILD_DIR/lib" -lopencv_dnn -lopencv_core -pthread -o probe
python3 -B cases.py
python3 -B measure.py verify --output verify_repeat.json
python3 -B measure.py bench --output bench_repeat.json
python3 -B measure.py bench --net --pairs 3 --cases ocr128,overlap --output net_repeat.json
```

Run the tests with the patched library:

```bash
export LD_LIBRARY_PATH="$OPENCV_BUILD_DIR/lib"
for opencv_threads in 1 4; do
  "$OPENCV_BUILD_DIR/bin/opencv_test_dnn" \
    --gtest_filter='*DeconvolutionCoordinates*:*Deconvolution.Accuracy*' \
    --test_threads="$opencv_threads"
done
taskset -c 0 "$OPENCV_BUILD_DIR/bin/opencv_perf_dnn" \
  --gtest_filter='*DeconvolutionCoordinates*' \
  --perf_min_samples=10 --perf_force_samples=30 --perf_threads=1
```

The accuracy test has 10 cases and uses an independent scatter reference.
Each thread count runs in a separate process. The cases cover 2D and 3D,
groups, static and dynamic weights, bias, overlap, dilation, asymmetric padding,
output padding, a shape change, and one or four threads. The separate probe
checks 192 configurations with normal and wide finite input values at both
thread counts: 768 comparisons and 5,417,400 output values.

For the extracted helper check against the previous PR version, run:

```bash
python3 -B check_col2im.py --baseline-ref 623c587 --wide-rows --output-dir helper-check
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ./helper-check/col2im_check
```

This check extracts both `Col2ImInvoker` implementations. It uses ASan and UBSan
for the extracted code and links the normal OpenCV core library. It is not a
sanitizer build of the full library. The 192 cases cover 1D, 2D, and 3D, special
float values, and output guard values. They also check that the column input
does not change. The current results at O1 and O3 are in
`run_reuse_20260925/sanitizer.json` and `sanitizer_o3.json`. The O3 check uses the
same source and compiler flags as the script, with `-O3` in place of `-O1`.
The previous patch also has native and forced scalar checks in
`row_simd_20260925/sanitizer_final.json` and `sanitizer_scalar.json`. Their source files are in
`row_simd_20260925/repro/final/`. The scalar source sets `CV_SIMD128` to zero. The earlier checks
also tested special bias values and the C++ vector implementation.
`CV_FORCE_SIMD128_CPP` does not disable the SIMD branch.

Nontrivial 1D full-layer cases failed on the base during test development.
The failure is in the existing 1D shape and workspace handling. This patch does
not change that code. Full-layer accuracy and performance claims are limited
to 2D and 3D. The extracted helper check also covers 1D.
