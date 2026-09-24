from pathlib import Path
import argparse
import os
import subprocess

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent.parent
REPO = Path(os.environ.get("OPENCV_SOURCE_DIR", WORKSPACE / ".tools/opencv-col2im"))
BUILD = Path(os.environ.get("OPENCV_BUILD_DIR", WORKSPACE / ".tools/opencv-col2im-build"))

DRIVER = r'''
int main()
{
    std::mt19937 random(20260920);
    size_t cases = 0, values = 0;
    for (int dims = 1; dims <= 3; ++dims)
        for (int test = 0; test < 32; ++test)
        {
            const int channels = 1 + random() % 3;
            const bool pointwise = test % 8 == 0;
            std::vector<int> input(dims), output(dims), kernel(dims), pads(dims);
            std::vector<int> stride(dims), dilation(dims);
            int inputSize = 1, outputSize = channels, kernelSize = channels;
            for (int d = 0; d < dims; ++d)
            {
                input[d] = 2 + random() % 3;
                kernel[d] = pointwise ? 1 : 1 + random() % 3;
                stride[d] = pointwise ? 1 : 1 + random() % 3;
                dilation[d] = pointwise ? 1 : 1 + random() % 2;
                pads[d] = pointwise ? 0 : random() % 2;
                output[d] = (input[d] - 1) * stride[d] + dilation[d] * (kernel[d] - 1)
                            + 1 - 2 * pads[d] + (pointwise ? 0 : random() % stride[d]);
                inputSize *= input[d];
                outputSize *= output[d];
                kernelSize *= kernel[d];
            }
            std::vector<float> column(inputSize * kernelSize), bias(channels);
            const uint32_t special[] = {0, 0x80000000u, 0x7f800000u, 0xff800000u,
                                         0x7fc01234u, 0x00000001u, 0x80000001u};
            for (size_t i = 0; i < column.size(); ++i)
            {
                column[i] = (int(random() % 129) - 64) / 32.0f;
                if (test % 4 == 0 && i % 11 == 0)
                    std::memcpy(&column[i], &special[(i / 11) % 7], sizeof(float));
            }
            for (float& x : bias)
                x = (int(random() % 33) - 16) / 8.0f;
            const auto saved = column;
            for (int threads : {1, 4})
            {
                cv::setNumThreads(threads);
                std::vector<float> before(outputSize + 16, 12345.0f), after = before;
                if (pointwise)
                {
                    std::copy(column.begin(), column.end(), before.begin() + 8);
                    after = before;
                }
                baseline::Col2ImInvoker::run(column.data(), channels, output, kernel,
                    pads, stride, dilation, input, before.data() + 8, bias.data(), pointwise);
                candidate::Col2ImInvoker::run(column.data(), channels, output, kernel,
                    pads, stride, dilation, input, after.data() + 8, bias.data(), pointwise);
                if (std::memcmp(before.data(), after.data(), before.size() * sizeof(float)) != 0 ||
                    std::memcmp(column.data(), saved.data(), column.size() * sizeof(float)) != 0)
                {
                    std::cerr << "Output differs: dims=" << dims << " case=" << test
                              << " threads=" << threads << '\n';
                    return 1;
                }
                for (int i = 0; i < 8; ++i)
                    if (after[i] != 12345.0f || after[outputSize + 8 + i] != 12345.0f)
                        return 2;
                ++cases;
                values += outputSize;
            }
        }
    std::cout << "{\"cases\":" << cases << ",\"values\":" << values
              << ",\"bitwise_equal\":true,\"guards_intact\":true}" << std::endl;
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    text = """// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include <opencv2/core.hpp>
#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
using cv::Range;
using cv::getNumThreads;
using cv::parallel_for_;
"""
    for namespace, path in [("baseline", ROOT / "deconvolution_before.cpp"),
                            ("candidate", REPO / "modules/dnn/src/layers/deconvolution_layer.cpp")]:
        code = path.read_text()
        start = code.index("    class Col2ImInvoker")
        end = code.index("\n#ifdef HAVE_OPENCL", start)
        text += "namespace " + namespace + " {\n" + code[start:end] + "\n}\n"
    text += DRIVER
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "col2im_check.cpp"
    source.write_text(text)
    if args.prepare_only:
        return
    command = ["g++", "-std=c++17", "-O1", "-g", "-fsanitize=address,undefined",
               "-fno-sanitize-recover=all", "-fno-omit-frame-pointer", "-no-pie", str(source),
               "-I" + str(REPO / "modules/core/include"), "-I" + str(BUILD),
               "-L" + str(BUILD / "lib"), "-lopencv_core", "-pthread",
               "-o", str(args.output_dir / "col2im_check")]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
