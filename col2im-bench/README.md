# CPU Deconvolution measurements

The base is OpenCV `5.x` at `8d126c4a672936fde5a21d9133c352b18d4299a5`.
The current patch is `ae8e518fdf49c9a8931cc2300707b2976d1cde09`.
The patch changes the CPU `Deconvolution` layer. The new ONNX `ConvTranspose2`
layer uses a different implementation. These measurements do not describe that
layer or a complete model.

The test host is an Intel Core Ultra 7 255H with Ubuntu 24.04 under WSL2.
The compiler is GCC 13.3.0. The build uses Release mode, SSE3 with AVX2 dispatch,
and the pthreads backend. IPP, OpenCL, and external BLAS libraries are disabled.

`review_20260924/bench_base.json` has five pairs of separate processes per case. Each process
has two warmup calls and 15 measured calls. The order of the two versions
alternates between pairs. Each reported time is a median of the process medians.
Each reduction is a median of the five paired reductions. The minimum and
maximum paired reductions are also included. CPU affinity is 0 for one thread,
and 0 through 3 for four threads. The four-thread results have more variation.
The host is not an isolated benchmark server. No build ran during measurement.

The measured `forward()` call includes GEMM, tap table setup, output reconstruction, and bias.
Layer setup and memory allocation for the input, weights, and output are outside
the timer. `review_20260924/bench_net.json` uses a single-layer `Net`. It sets the input
again before each measured forward call. The input setup is outside the timer.
Both probes compare every output byte between the two versions.

`review_20260924/bench_previous.json` compares the previous PR commit `eafeffd8`
with the current patch. It uses the same cases and settings. For small inputs
with four threads, the first base comparison has slower pairs. The repeat and
same-binary control both have large variation. Those results are in
`review_20260924/bench_tiny_repeat.json` and `review_20260924/bench_tiny_control.json`.
Do not use these small cases to claim a stable speedup or no regression.
The older `bench_final.json` describes the previous implementation.

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

For the extracted helper check, save the base source as
`deconvolution_before.cpp` in this directory, then run:

```bash
python3 -B check_col2im.py
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ./col2im_check
```

This check extracts both `Col2ImInvoker` implementations. It uses ASan and UBSan
for the extracted code and links the normal OpenCV core library. It is not a
sanitizer build of the full library. The 192 cases cover 1D, 2D, and 3D, special
float values, and output guard values. They also check that the column input
does not change.

Nontrivial 1D full-layer cases failed on the base during test development.
The failure is in the existing 1D shape and workspace handling. This patch does
not change that code. Full-layer accuracy and performance claims are limited
to 2D and 3D. The extracted helper check also covers 1D.

