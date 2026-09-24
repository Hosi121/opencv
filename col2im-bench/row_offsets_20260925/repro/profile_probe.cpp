#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <algorithm>
#include <array>
#include <dlfcn.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace cv;
using namespace cv::dnn;

static void fill(Mat& value, unsigned seed, bool wide)
{
    uint32_t state = seed;
    for (size_t i = 0; i < value.total(); ++i)
    {
        state = state * 1664525u + 1013904223u;
        float x = (int((state >> 16) % 129) - 64) / 128.0f;
        value.ptr<float>()[i] = wide ? std::ldexp(x, int(state % 51) - 25) : x;
    }
}

int main(int argc, char** argv)
{
    if (argc < 6)
    {
        std::cerr << "Use: probe cases.txt name-or-all threads iterations output.bin\n";
        return 1;
    }
    setNumThreads(std::atoi(argv[3]));
    const int iterations = std::atoi(argv[4]);
    const bool wide = std::getenv("PROBE_WIDE_VALUES") != nullptr;
    const bool useNet = std::getenv("PROBE_NET") != nullptr;
    const auto get_profile = reinterpret_cast<const int64* (*)()>(dlsym(RTLD_DEFAULT, "opencv_deconv_profile"));
    const double tick_to_ms = 1000.0 / getTickFrequency();
    std::ifstream cases(argv[1]);
    std::ofstream dump(argv[5], std::ios::binary);
    if (!cases || !dump || iterations < 1)
        return 2;
    std::string line;
    int matched = 0;
    while (std::getline(cases, line))
    {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream row(line);
        std::string name;
        int dims, batch, ci, co, groups, hasBias, dynamic;
        if (!(row >> name >> dims >> batch >> ci >> co >> groups >> hasBias >> dynamic))
            return 3;
        if (std::string(argv[2]) != "all" && name != argv[2])
            continue;
        ++matched;
        std::vector<int> spatial(dims), kernel(dims), stride(dims), dilation(dims);
        std::vector<int> pads(dims * 2), adjust(dims);
        for (auto* v : {&spatial, &kernel, &stride, &dilation, &pads, &adjust})
            for (int& x : *v)
                if (!(row >> x))
                    return 4;
        std::vector<int> inputShape = {batch, ci}, weightShape = {ci, co / groups};
        inputShape.insert(inputShape.end(), spatial.begin(), spatial.end());
        weightShape.insert(weightShape.end(), kernel.begin(), kernel.end());
        Mat input(inputShape, CV_32F), weights(weightShape, CV_32F), bias(1, co, CV_32F);
        fill(input, 17, wide);
        fill(weights, 29, wide);
        fill(bias, 43, wide);
        LayerParams lp;
        lp.set("kernel_size", DictValue::arrayInt(kernel.data(), dims));
        lp.set("stride", DictValue::arrayInt(stride.data(), dims));
        lp.set("dilation", DictValue::arrayInt(dilation.data(), dims));
        lp.set("pad", DictValue::arrayInt(pads.data(), 2 * dims));
        lp.set("adj", DictValue::arrayInt(adjust.data(), dims));
        lp.set("num_output", co);
        lp.set("group", groups);
        lp.set("bias_term", bool(hasBias));
        if (!dynamic)
        {
            lp.blobs.push_back(weights);
            if (hasBias)
                lp.blobs.push_back(bias);
        }
        std::vector<Mat> inputs(1, input), outputs, internals;
        if (dynamic)
        {
            inputs.push_back(weights);
            if (hasBias)
                inputs.push_back(bias);
        }
        Ptr<Layer> layer = DeconvolutionLayer::create(lp);
        std::vector<MatShape> inputShapes, outputShapes, internalShapes;
        for (const Mat& x : inputs)
            inputShapes.push_back(x.shape());
        layer->getMemoryShapes(inputShapes, 0, outputShapes, internalShapes);
        for (const MatShape& shape : outputShapes)
            outputs.push_back(Mat(shape, CV_32F));
        for (const MatShape& shape : internalShapes)
            internals.push_back(Mat(shape, CV_32F));
        layer->finalize(inputs, outputs);
        Net net;
        if (useNet)
        {
            if (dynamic)
                return 5;
            net.addLayerToPrev("deconv", "Deconvolution", lp);
            net.setPreferableBackend(DNN_BACKEND_OPENCV);
            net.setPreferableTarget(DNN_TARGET_CPU);
            net.setInput(input);
        }
        auto run = [&]() {
            if (useNet)
                outputs[0] = net.forward();
            else
                layer->forward(inputs, outputs, internals);
        };
        run();
        run();
        std::vector<double> times;
        std::vector<std::array<double, 6> > phases;
        phases.reserve(iterations);
        for (int i = 0; i < iterations; ++i)
        {
            if (useNet)
                net.setInput(input);
            auto start = std::chrono::steady_clock::now();
            run();
            times.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
            if (get_profile)
            {
                const int64* ticks = get_profile();
                phases.push_back({ticks[0] * tick_to_ms, ticks[1] * tick_to_ms,
                                  ticks[2] * tick_to_ms, ticks[3] * tick_to_ms, times.back(), double(ticks[4])});
            }
        }
        for (const auto& p : phases)
            std::cerr << std::setprecision(12) << "DECONV_PHASE {\"gemm_ms\":" << p[0]
                      << ",\"tap_setup_ms\":" << p[1] << ",\"run_setup_ms\":" << p[2]
                      << ",\"output_ms\":" << p[3] << ",\"total_ms\":" << p[4] << ",\"workspace_bytes\":" << p[5] << "}\n";
        std::sort(times.begin(), times.end());
        const Mat& out = outputs[0];
        if (!out.isContinuous() || !checkRange(out))
            return 6;
        uint64_t hash = 14695981039346656037ull;
        for (size_t i = 0; i < out.total() * sizeof(float); ++i)
        {
            hash ^= out.data[i];
            hash *= 1099511628211ull;
        }
        dump.write(reinterpret_cast<const char*>(out.data), out.total() * sizeof(float));
        std::cout << std::setprecision(10) << "{\"case\":\"" << name << "\",\"threads\":" << getNumThreads()
                  << ",\"median_ms\":" << times[times.size() / 2] << ",\"min_ms\":" << times.front()
                  << ",\"iterations\":" << iterations << ",\"elements\":" << out.total()
                  << ",\"hash\":\"" << hash << "\",\"net\":" << (useNet ? "true" : "false") << "}\n";
    }
    return matched ? 0 : 7;
}
