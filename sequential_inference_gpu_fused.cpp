/*
Compile:
nvcc -std=c++17 -O2 sequential_inference_gpu_fused_min.cpp -o run_trt_gpu_fused_min \
  -I/usr/include/x86_64-linux-gnu \
  -I/usr/local/cuda/include \
  -L/usr/lib/x86_64-linux-gnu \
  -L/usr/local/cuda/lib64 \
  -lnvinfer -lnvonnxparser -lcudart
*/

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(x)                                                   \
    do {                                                                \
        cudaError_t err = (x);                                          \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)      \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)

struct Logger : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << "\n";
    }
};

static size_t volume(const nvinfer1::Dims& d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i)
        v *= static_cast<size_t>(d.d[i]);
    return v;
}

static std::vector<char> loadFile(const fs::path& p) {
    std::ifstream ifs(p, std::ios::binary | std::ios::ate);
    if (!ifs) throw std::runtime_error("Failed to open: " + p.string());
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    if (!ifs.read(buf.data(), size))
        throw std::runtime_error("Failed to read: " + p.string());
    return buf;
}

int main() {
    try {
        Logger logger;
        auto runtime = std::unique_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(logger));

        // ------------------------------------------------------------
        // 1. Load all chunk engines and create contexts
        // ------------------------------------------------------------
        std::vector<fs::path> enginePaths;
        for (auto& e : fs::directory_iterator("engines"))
            if (e.path().extension() == ".engine")
                enginePaths.push_back(e.path());
        std::sort(enginePaths.begin(), enginePaths.end());
        if (enginePaths.empty())
            throw std::runtime_error("No .engine files found in ./engines");

        std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> engines;
        std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> contexts;
        for (auto& p : enginePaths) {
            auto bytes = loadFile(p);
            auto* e = runtime->deserializeCudaEngine(bytes.data(), bytes.size());
            if (!e)
                throw std::runtime_error("Failed to load engine: " + p.string());
            engines.emplace_back(e);
            contexts.emplace_back(e->createExecutionContext());
            std::cout << "[INFO] Loaded engine: " << p.filename().string() << "\n";
        }

        int numChunks = engines.size();
        std::cout << "✅ Loaded " << numChunks << " chunk engines.\n";

        // ------------------------------------------------------------
        // 2. Determine buffer sizes and total GPU memory needed
        // ------------------------------------------------------------
        const int B = 1, C = 3, H = 224, W = 224;
        nvinfer1::Dims inputDims{4, {B, C, H, W}};
        std::vector<nvinfer1::Dims> inDims(numChunks), outDims(numChunks);

        size_t totalBytes = 0;
        std::vector<size_t> offsets(numChunks + 1);

        for (int i = 0; i < numChunks; ++i) {
            const char* inName = engines[i]->getIOTensorName(0);
            const char* outName = engines[i]->getIOTensorName(1);
            inDims[i] = engines[i]->getTensorShape(inName);
            outDims[i] = engines[i]->getTensorShape(outName);

            if (std::any_of(inDims[i].d, inDims[i].d + inDims[i].nbDims,
                            [](int d) { return d < 0; }))
                inDims[i] = inputDims;
            contexts[i]->setInputShape(inName, inDims[i]);

            offsets[i] = totalBytes;
            totalBytes += volume(outDims[i]) * sizeof(float);
        }
        offsets[numChunks] = totalBytes;

        size_t inputBytes = volume(inputDims) * sizeof(float);
        totalBytes += inputBytes;

        std::cout << "[INFO] Total fused GPU memory = "
                  << (double)totalBytes / (1024 * 1024) << " MB\n";

        // ------------------------------------------------------------
        // 3. Allocate one big device buffer
        // ------------------------------------------------------------
        void* dBase = nullptr;
        CUDA_CHECK(cudaMalloc(&dBase, totalBytes));
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* dInput = dBase;
        std::vector<void*> dOut(numChunks);
        size_t currentOffset = inputBytes;
        for (int i = 0; i < numChunks; ++i) {
            dOut[i] = static_cast<void*>(static_cast<char*>(dBase) + currentOffset);
            currentOffset += volume(outDims[i]) * sizeof(float);
        }

        // ------------------------------------------------------------
        // 4. Upload input once
        // ------------------------------------------------------------
        std::vector<float> hInput(volume(inputDims));
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.f, 1.f);
        for (auto& v : hInput) v = dist(rng);
        CUDA_CHECK(cudaMemcpyAsync(dInput, hInput.data(),
                                   inputBytes, cudaMemcpyHostToDevice, stream));

        // ------------------------------------------------------------
        // 5. Pre-bind tensor addresses once
        // ------------------------------------------------------------
        void* dSrc = dInput;
        for (int i = 0; i < numChunks; ++i) {
            const char* inName = engines[i]->getIOTensorName(0);
            const char* outName = engines[i]->getIOTensorName(1);
            contexts[i]->setTensorAddress(inName, dSrc);
            contexts[i]->setTensorAddress(outName, dOut[i]);
            dSrc = dOut[i];
        }

        // ------------------------------------------------------------
        // 6. Run sequentially on GPU only (minimal overhead)
        // ------------------------------------------------------------
        const int WARMUPS = 10;
        const int RUNS = 100;
        std::vector<float> gpuTotals(RUNS);
        std::vector<float> cpuTotals(RUNS);

        auto runPass = [&](int r, bool record) {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventRecord(start, stream));

            // Chained enqueue (no pointer resets or host syncs)
            for (int i = 0; i < numChunks; ++i)
                contexts[i]->enqueueV3(stream);

            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            auto t1 = std::chrono::high_resolution_clock::now();

            float gpuMs = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&gpuMs, start, stop));
            double cpuMs =
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            if (record) {
                gpuTotals[r] = gpuMs;
                cpuTotals[r] = cpuMs;
            }
        };

        for (int i = 0; i < WARMUPS; ++i) runPass(i, false);
        for (int r = 0; r < RUNS; ++r) runPass(r, true);

        // ------------------------------------------------------------
        // 7. Save timing results
        // ------------------------------------------------------------
        std::ofstream f("chunk_timing.csv");
        f << "run_id,total_gpu_ms,total_cpu_ms\n";
        for (int r = 0; r < RUNS; ++r)
            f << (r + 1) << ","
              << std::fixed << std::setprecision(6)
              << gpuTotals[r] << "," << cpuTotals[r] << "\n";
        f.close();
        std::cout << "📝 Saved results to chunk_timing.csv\n";

        // ------------------------------------------------------------
        // 8. Cleanup
        // ------------------------------------------------------------
        cudaFree(dBase);
        cudaStreamDestroy(stream);
        std::cout << "✅ Done.\n";

    } catch (std::exception const& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
