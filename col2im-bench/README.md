# CPU Deconvolution measurements

The fixed upstream base is `8d126c4a672936fde5a21d9133c352b18d4299a5`.
The current patch is `d0d1825d5a91cfffe51550196bfee7896d60c2a3`. The previous PR commit is `032ebee`.
The patch changes the legacy CPU `Deconvolution` layer. The new ONNX
`ConvTranspose2` layer has a separate implementation. These are layer results.
Complete models and ARM hardware were not tested.

The code shares valid tap tables across channels and workers. It computes outer
tap combinations once per row and input addresses once per SIMD run. A 2D row
uses the outer-axis table directly. The SIMD path uses horizontal stride 1 or 2.
Other positions use scalar code. Each output keeps the original sum order.

The host is an Intel Core Ultra 7 255H with Ubuntu 24.04 under WSL2. The compiler
is GCC 13.3.0. The build uses Release mode, SSE3 with AVX2 dispatch, and pthreads.
IPP, OpenCL, and external BLAS libraries are disabled.

## Results and evidence

One-thread layer results against the fixed upstream base:

| Input, channels in/out | Kernel, stride | Upstream | PR | Speedup |
| --- | --- | ---: | ---: | ---: |
| 128×128, 16/16 | 1×1, 1 | 3.023 ms | 0.089 ms | 34.2× |
| 128×128, 16/16 | 2×2, 2 | 76.145 ms | 0.489 ms | 155.9× |
| 64×64, 16/16 | 3×3, 1, pad 1 | 10.709 ms | 0.239 ms | 44.8× |
| 16×16×16, 8/8 | 3×3×3, 1 | 21.843 ms | 0.472 ms | 46.3× |

This comparison uses seven pairs of separate processes, with two warmup calls
and 31 measured calls per process. Version order alternates. Times are medians
of process medians. Speedup uses the unrounded times. CPU affinity is 0 for one
thread and 0 through 3 for four threads. The host is shared. No build ran during
measurement. Some small-input and four-thread pairs are slower; the raw data
retains these samples.

The current evidence is in `row_offsets_20260925/`:

| File | Purpose |
| --- | --- |
| `build_source.json` | Source, build, scope, and test review |
| `bench_base_final.json` | Fixed upstream to the final patch; seven pairs, 31 calls |
| `bench_final.json` | Previous PR to the final patch; seven pairs, 101 calls, one and four threads |
| `bench_large.json` | Larger 3x3 inputs, channels, stride, and dilation; five pairs, 101 calls |
| `verify_final.json` | 768 output comparisons with upstream |
| `profile_final.json`, `profile_final_summary.json` | Diagnostic phase times and worker buffer capacities |
| `trials.json`, `repro/` | Earlier trials, patches, and reasons to keep or discard each change |

The layer timer includes GEMM, table setup, output reconstruction, and bias.
Layer setup and input, weight, and output allocation are outside the timer. The
Net probe sets the input before each timed call. Both probes compare the outputs.

The final 3x3 profile assigns about 69 percent of time to GEMM and 30 percent to
output reconstruction. The worker buffer capacities are 16 bytes for the 2x2
case, 96 bytes for 3x3, and 384 bytes for 3D. These one-thread measurements exclude
vector objects, shared tables, and allocator overhead. The shared SIMD run table
uses one `size_t` per output column. It is 2,048 bytes for the 128-square 2x2 case.
The diagnostic probe reads counters after the layer timer stops and writes logs
after all iterations. Tiny-input timings have substantial timer overhead.

Older results are in `row_taps_20260925/` (`032ebee`), `run_reuse_20260925/`
(`27dadfe`), `row_simd_20260925/` (`623c587`), and `review_20260924/` (`ae8e518`).
Their source records and measurements remain available. They are not measurements
of the current patch.

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
python3 -B measure.py bench --pairs 7 --iterations 31 --threads 1 \
  --cases pointwise,ocr128,overlap,three_d --output bench_repeat.json
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

The accuracy test has 8 cases and uses an independent scatter reference.
Each thread count runs in a separate process. The cases cover 2D and 3D,
groups, static and dynamic weights, bias, overlap, dilation, asymmetric padding,
output padding, a shape change, and one or four threads. The separate probe
checks 192 configurations with normal and wide finite input values at both
thread counts: 768 comparisons and 5,417,400 output values.

For the extracted helper check against the previous PR commit, run:

```bash
python3 -B check_col2im.py --baseline-ref 032ebee --wide-rows --max-dims 4 --output-dir helper-check
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ./helper-check/col2im_check
```

This check uses ASan and UBSan for the extracted helper and links the normal
OpenCV core library. It is not a sanitizer build of the full library. Its 256
cases cover one through four spatial dimensions, special float values, output
guards, and unchanged column input. The saved final sources are in
`row_offsets_20260925/repro/final/`. The O3 check uses the same source and compiler
flags, with `-O3` in place of `-O1`. `scalar_check.cpp` sets `CV_SIMD128` to zero
after the intrinsic headers. It uses the same compiler and sanitizer flags.

All 56 accuracy tests (8 new and 48 existing) pass at one and four threads with
the current library and with the upstream library. The six performance tests
pass. The helper checks at O1, O3, and with forced scalar code pass.

Nontrivial 1D full-layer cases failed on the base during test development. The
failure is in existing shape and workspace handling. Full-layer claims here
are limited to 2D and 3D. The extracted helper also covers 1D and 4D.

## Restore the local files

In the original `/home/hosi/churin` workspace, restore these archives in order:

```bash
tar -xJf archive/opencv_deconv_20260920/artifacts.tar.xz -C /home/hosi/churin
tar -xJf archive/opencv_deconv_20260920/test_cleanup_20260924.tar.xz -C /home/hosi/churin
tar -xJf archive/opencv_deconv_20260920/review_20260924.tar.xz -C /home/hosi/churin
tar -xJf archive/opencv_deconv_20260920/row_simd_20260925.tar.xz -C /home/hosi/churin
tar -xJf archive/opencv_deconv_20260920/run_reuse_20260925.tar.xz -C /home/hosi/churin
tar -xJf archive/opencv_deconv_20260920/row_taps_20260925.tar.xz -C /home/hosi/churin
tar -xJf archive/opencv_deconv_20260920/row_offsets_20260925.tar.xz -C /home/hosi/churin
```

To restore `032ebee` as a comparison library, extract only
`.tools/opencv-col2im-build/lib/libopencv_dnn.so.5.1.0` from the row taps archive
into `OPENCV_BEFORE_DIR`. Keep its name and add `libopencv_dnn.so.501` as a symlink
to it. This avoids another archived copy.

For a new build in this workspace, restore the sparse source files:

```bash
git -C .tools/opencv-col2im sparse-checkout set --no-cone --stdin \
  < results/oss/opencv_deconv_20260920/source_sparse_for_build.txt
git -C .tools/opencv-col2im restore --ignore-skip-worktree-bits --source=HEAD -- \
  doc/CMakeLists.txt docs_sphinx/CMakeLists.txt
cmake -S .tools/opencv-col2im -B .tools/opencv-col2im-build
```
