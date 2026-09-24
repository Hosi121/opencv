// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
using cv::Range;
using cv::getNumThreads;
using cv::parallel_for_;
using namespace cv;
namespace baseline {
    class Col2ImInvoker : public cv::ParallelLoopBody
    {
    public:
        const float* data_col;
        const float* biasvec;
        int channels;
        std::vector<int> output_shape;  // spatial dimensions only
        std::vector<std::vector<size_t> > tap_starts;
        std::vector<std::vector<size_t> > tap_offsets;
        size_t output_spatial_size;
        size_t column_channel_stride;
        float* data_im;
        int nstripes;
        bool is1x1;

        Col2ImInvoker()
            : data_col(0), biasvec(0), channels(0), output_spatial_size(1),
              column_channel_stride(0), data_im(0),
              nstripes(0), is1x1(0)
        {}

        static void run(const float* data_col,
                        int channels,
                        const std::vector<int>& output_shape,
                        const std::vector<int>& kernel_shape,
                        const std::vector<int>& pads,
                        const std::vector<int>& strides,
                        const std::vector<int>& dilations,
                        const std::vector<int>& input_shape,
                        float* data_im,
                        const float* biasvec,
                        bool is1x1)
        {
            const int nstripes = getNumThreads();

            Col2ImInvoker t;
            t.data_col = data_col;
            t.data_im = data_im;
            t.channels = channels;
            t.output_shape = output_shape;
            t.nstripes = nstripes;
            t.is1x1 = is1x1;
            t.biasvec = biasvec;

            const int ndims = output_shape.size();
            for (int d = 0; d < ndims; d++)
                t.output_spatial_size *= output_shape[d];

            if (!is1x1)
            {
                size_t kernel_step = 1, input_step = 1;
                for (int d = 0; d < ndims; d++)
                    kernel_step *= input_shape[d];

                // Store the kernel and input offset for each valid tap on each axis.
                // The tables are shared by all channels and workers.
                t.tap_starts.resize(ndims);
                t.tap_offsets.resize(ndims);
                for (int d = ndims - 1; d >= 0; d--)
                {
                    std::vector<size_t>& starts = t.tap_starts[d];
                    std::vector<size_t>& offsets = t.tap_offsets[d];
                    starts.resize(output_shape[d] + 1);
                    for (int out = 0; out < output_shape[d]; out++)
                    {
                        starts[out] = offsets.size();
                        for (int k = 0; k < kernel_shape[d]; k++)
                        {
                            int input = out + pads[d] - k * dilations[d];
                            if (input < 0 || input % strides[d] != 0)
                                continue;
                            input /= strides[d];
                            if (input < input_shape[d])
                                offsets.push_back(k * kernel_step + input * input_step);
                        }
                    }
                    starts[output_shape[d]] = offsets.size();
                    kernel_step *= kernel_shape[d];
                    input_step *= input_shape[d];
                }
                t.column_channel_stride = kernel_step;
            }

            parallel_for_(Range(0, nstripes), t, nstripes);
        }

        virtual void operator ()(const Range &r) const CV_OVERRIDE
        {
            const float* data_col_ = data_col;
            float* data_im_ = data_im;
            bool is1x1_ = is1x1;
            const float* biasvec_ = biasvec;

            int ndims = output_shape.size();

            const size_t total_output_size = channels * output_spatial_size;

            size_t stripeSize = (total_output_size + nstripes - 1) / nstripes;
            size_t startIndex = r.start * stripeSize;
            size_t endIndex = std::min(r.end * stripeSize, (size_t)total_output_size);

            if (startIndex >= endIndex)
                return;

            if (is1x1_)
            {
                size_t index = startIndex;
                while (index < endIndex)
                {
                    const size_t channel = index / output_spatial_size;
                    const size_t channel_end = std::min((channel + 1) * output_spatial_size, endIndex);
                    const float bias = biasvec_[channel];
                    for (; index < channel_end; index++)
                        data_im_[index] += bias;
                }
                return;
            }

            std::vector<size_t> tap_begin(ndims), tap_end(ndims), tap_index(ndims);

            for (size_t index = startIndex; index < endIndex; index++)
            {
                size_t idx = index;
                size_t col_offset = 0;
                bool has_taps = true;
                for (int d = ndims - 1; d >= 0; d--)
                {
                    const size_t coord = idx % output_shape[d];
                    idx /= output_shape[d];
                    tap_index[d] = tap_begin[d] = tap_starts[d][coord];
                    tap_end[d] = tap_starts[d][coord + 1];
                    if (tap_begin[d] == tap_end[d])
                        has_taps = false;
                    else
                        col_offset += tap_offsets[d][tap_begin[d]];
                }
                col_offset += idx * column_channel_stride;

                float val = 0.0f;
                if (has_taps)
                {
                    // Visit valid taps in the same order as the original nested kernel loops.
                    for (;;)
                    {
                        val += data_col_[col_offset];
                        int d = ndims - 1;
                        for (; d >= 0; d--)
                        {
                            col_offset -= tap_offsets[d][tap_index[d]];
                            if (++tap_index[d] < tap_end[d])
                            {
                                col_offset += tap_offsets[d][tap_index[d]];
                                break;
                            }
                            tap_index[d] = tap_begin[d];
                            col_offset += tap_offsets[d][tap_index[d]];
                        }
                        if (d < 0)
                            break;
                    }
                }
                data_im_[index] = val + biasvec_[idx];
            }
        }
    };

}
namespace candidate {
    class Col2ImInvoker : public cv::ParallelLoopBody
    {
    public:
        const float* data_col;
        const float* biasvec;
        int channels;
        std::vector<int> output_shape;  // spatial dimensions only
        std::vector<std::vector<size_t> > tap_starts;
        std::vector<std::vector<size_t> > tap_offsets;
        std::vector<size_t> x_run_end;
        int x_stride;
        size_t output_spatial_size;
        size_t column_channel_stride;
        float* data_im;
        int nstripes;
        bool is1x1;

        Col2ImInvoker()
            : data_col(0), biasvec(0), channels(0), x_stride(0), output_spatial_size(1),
              column_channel_stride(0), data_im(0),
              nstripes(0), is1x1(0)
        {}

        static void run(const float* data_col,
                        int channels,
                        const std::vector<int>& output_shape,
                        const std::vector<int>& kernel_shape,
                        const std::vector<int>& pads,
                        const std::vector<int>& strides,
                        const std::vector<int>& dilations,
                        const std::vector<int>& input_shape,
                        float* data_im,
                        const float* biasvec,
                        bool is1x1)
        {
            const int nstripes = getNumThreads();

            Col2ImInvoker t;
            t.data_col = data_col;
            t.data_im = data_im;
            t.channels = channels;
            t.output_shape = output_shape;
            t.nstripes = nstripes;
            t.is1x1 = is1x1;
            t.biasvec = biasvec;

            const int ndims = output_shape.size();
            for (int d = 0; d < ndims; d++)
                t.output_spatial_size *= output_shape[d];

            if (!is1x1)
            {
                size_t kernel_step = 1, input_step = 1;
                for (int d = 0; d < ndims; d++)
                    kernel_step *= input_shape[d];

                // Store the kernel and input offset for each valid tap on each axis.
                // The tables are shared by all channels and workers.
                t.tap_starts.resize(ndims);
                t.tap_offsets.resize(ndims);
                for (int d = ndims - 1; d >= 0; d--)
                {
                    std::vector<size_t>& starts = t.tap_starts[d];
                    std::vector<size_t>& offsets = t.tap_offsets[d];
                    starts.resize(output_shape[d] + 1);
                    for (int out = 0; out < output_shape[d]; out++)
                    {
                        starts[out] = offsets.size();
                        for (int k = 0; k < kernel_shape[d]; k++)
                        {
                            int input = out + pads[d] - k * dilations[d];
                            if (input < 0 || input % strides[d] != 0)
                                continue;
                            input /= strides[d];
                            if (input < input_shape[d])
                                offsets.push_back(k * kernel_step + input * input_step);
                        }
                    }
                    starts[output_shape[d]] = offsets.size();
                    kernel_step *= kernel_shape[d];
                    input_step *= input_shape[d];
                }
                t.column_channel_stride = kernel_step;
#if CV_SIMD128
                const int last = ndims - 1;
                const size_t width = output_shape[last];
                const int step = strides[last];
                if (step <= 2 && width >= (size_t)(4 * step))
                {
                    t.x_stride = step;
                    t.x_run_end.resize(width);
                    const std::vector<size_t>& starts = t.tap_starts[last];
                    const std::vector<size_t>& offsets = t.tap_offsets[last];
                    for (size_t x = width; x-- > 0;)
                    {
                        const size_t next = x + step;
                        t.x_run_end[x] = next;
                        if (next >= width)
                            continue;
                        const size_t begin = starts[x], next_begin = starts[next];
                        const size_t count = starts[x + 1] - begin;
                        if (starts[next + 1] - next_begin != count)
                            continue;
                        size_t k = 0;
                        for (; k < count; k++)
                            if (offsets[next_begin + k] != offsets[begin + k] + 1)
                                break;
                        if (k == count)
                            t.x_run_end[x] = t.x_run_end[next];
                    }
                }
#endif
            }

            parallel_for_(Range(0, nstripes), t, nstripes);
        }

        bool next_tap(size_t& offset, std::vector<size_t>& index,
                      const std::vector<size_t>& begin, const std::vector<size_t>& end,
                      int last) const
        {
            for (int d = last; d >= 0; d--)
            {
                offset -= tap_offsets[d][index[d]];
                if (++index[d] < end[d])
                {
                    offset += tap_offsets[d][index[d]];
                    return true;
                }
                index[d] = begin[d];
                offset += tap_offsets[d][index[d]];
            }
            return false;
        }

        virtual void operator ()(const Range &r) const CV_OVERRIDE
        {
            const float* data_col_ = data_col;
            float* data_im_ = data_im;
            bool is1x1_ = is1x1;
            const float* biasvec_ = biasvec;

            int ndims = output_shape.size();

            const size_t total_output_size = channels * output_spatial_size;

            size_t stripeSize = (total_output_size + nstripes - 1) / nstripes;
            size_t startIndex = r.start * stripeSize;
            size_t endIndex = std::min(r.end * stripeSize, (size_t)total_output_size);

            if (startIndex >= endIndex)
                return;

            if (is1x1_)
            {
                size_t index = startIndex;
                while (index < endIndex)
                {
                    const size_t channel = index / output_spatial_size;
                    const size_t channel_end = std::min((channel + 1) * output_spatial_size, endIndex);
                    const float bias = biasvec_[channel];
                    for (; index < channel_end; index++)
                        data_im_[index] += bias;
                }
                return;
            }

            std::vector<size_t> tap_begin(ndims), tap_end(ndims), tap_index(ndims);
            const int last = ndims - 1;
            const size_t width = output_shape[last];
            const std::vector<size_t>& x_starts = tap_starts[last];
            const std::vector<size_t>& x_offsets = tap_offsets[last];
            size_t index = startIndex;
            for (size_t row = startIndex / width; index < endIndex; row++)
            {
                size_t idx = row;
                size_t row_offset = 0;
                bool has_outer_taps = true;
                for (int d = last - 1; d >= 0; d--)
                {
                    const size_t coord = idx % output_shape[d];
                    idx /= output_shape[d];
                    tap_index[d] = tap_begin[d] = tap_starts[d][coord];
                    tap_end[d] = tap_starts[d][coord + 1];
                    if (tap_begin[d] == tap_end[d])
                        has_outer_taps = false;
                    else
                        row_offset += tap_offsets[d][tap_begin[d]];
                }
                row_offset += idx * column_channel_stride;
                const float bias = biasvec_[idx];
                const size_t row_start = row * width;
                const size_t row_end = std::min(row_start + width, endIndex);

#if CV_SIMD128
                const v_float32x4 vbias = v_setall_f32(bias);
                auto sum_four = [&](size_t x) {
                    v_float32x4 val = v_setzero_f32();
                    if (has_outer_taps)
                    {
                        size_t offset = row_offset;
                        do
                        {
                            for (size_t k = x_starts[x]; k < x_starts[x + 1]; k++)
                                val = v_add(val, v_load(data_col_ + offset + x_offsets[k]));
                        } while (next_tap(offset, tap_index, tap_begin, tap_end, last - 1));
                    }
                    return v_add(val, vbias);
                };
#endif
                size_t x = index - row_start;
                while (index < row_end)
                {
#if CV_SIMD128
                    const size_t span = 4 * x_stride;
                    if (x_stride && index + span <= row_end && x_run_end[x] >= x + span &&
                        (x_stride == 1 || x_run_end[x + 1] >= x + 1 + span))
                    {
                        const v_float32x4 v0 = sum_four(x);
                        if (x_stride == 1)
                            v_store(data_im_ + index, v0);
                        else
                            v_store_interleave(data_im_ + index, v0, sum_four(x + 1));
                        x += span;
                        index += span;
                        continue;
                    }
#endif
                    tap_index[last] = tap_begin[last] = x_starts[x];
                    tap_end[last] = x_starts[x + 1];
                    float val = 0.0f;
                    if (has_outer_taps && tap_begin[last] != tap_end[last])
                    {
                        size_t col_offset = row_offset + x_offsets[tap_begin[last]];
                        // Keep the sum order. The last axis changes first.
                        do
                        {
                            val += data_col_[col_offset];
                        } while (next_tap(col_offset, tap_index, tap_begin, tap_end, last));
                    }
                    data_im_[index] = val + bias;
                    x++;
                    index++;
                }
            }
        }
    };

}

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
                input[d] = (d == dims - 1 && !pointwise && test % 4 == 0 ? 9 : 2) + random() % 3;
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
