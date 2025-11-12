/*
Compile:
nvcc -std=c++17 -O2 sequential_inference_naive.cpp -o run_trt_naive \
  -I/usr/include/x86_64-linux-gnu \
  -I/usr/local/cuda/include \
  -L/usr/lib/x86_64-linux-gnu \
  -L/usr/local/cuda/lib64 \
  -lnvinfer -lnvonnxparser -lcudart -lnvToolsExt
*/

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

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
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
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
        // 1. Load all engine chunks
        // ------------------------------------------------------------
        std::vector<fs::path> enginePaths;
        for (auto& e : fs::directory_iterator("engines"))
            if (e.path().extension() == ".engine")
                enginePaths.push_back(e.path());
        std::sort(enginePaths.begin(), enginePaths.end());
        if (enginePaths.empty())
            throw std::runtime_error("No .engine files found under ./engines");

        std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> engines;
        std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> contexts;

        for (auto& p : enginePaths) {
            auto bytes = loadFile(p);
            auto* e = runtime->deserializeCudaEngine(bytes.data(), bytes.size());
            if (!e) throw std::runtime_error("Failed to load engine: " + p.string());
            engines.emplace_back(e);
            contexts.emplace_back(e->createExecutionContext());
            std::cout << "[INFO] Loaded engine: " << p.filename().string() << "\n";
        }
        int numChunks = engines.size();
        std::cout << "✅ Loaded " << numChunks << " chunk engines.\n";

        // ------------------------------------------------------------
        // 2. Allocate GPU buffers
        // ------------------------------------------------------------
        const int B = 1, C = 3, H = 224, W = 224;
        nvinfer1::Dims inputDims{4, {B, C, H, W}};

        std::vector<nvinfer1::Dims> inDims(numChunks), outDims(numChunks);

        size_t maxBytes = 0;
        for (int i = 0; i < numChunks; ++i) {
            const char* inName = engines[i]->getIOTensorName(0);
            const char* outName = engines[i]->getIOTensorName(1);
            inDims[i] = engines[i]->getTensorShape(inName);
            outDims[i] = engines[i]->getTensorShape(outName);
            if (std::any_of(inDims[i].d, inDims[i].d + inDims[i].nbDims,
                            [](int d) { return d < 0; }))
                inDims[i] = inputDims;
            contexts[i]->setInputShape(inName, inDims[i]);
            maxBytes = std::max(maxBytes, volume(outDims[i]) * sizeof(float));
        }

        void* dInput = nullptr;
        void* dOutput = nullptr;
        CUDA_CHECK(cudaMalloc(&dInput, maxBytes));
        CUDA_CHECK(cudaMalloc(&dOutput, maxBytes));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::cout << "[INFO] Allocated " << (maxBytes / 1024.0 / 1024.0)
                  << " MB reusable GPU buffers.\n";

        // ------------------------------------------------------------
        // 3. Create random input
        // ------------------------------------------------------------
        std::vector<float> hInput(volume(inputDims));
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.f, 1.f);
        for (auto& v : hInput) v = dist(rng);

        // ------------------------------------------------------------
        // Helper: run one chunk
        // ------------------------------------------------------------
        auto runChunk = [&](nvinfer1::IExecutionContext* ctx,
                            std::vector<float>& hIn,
                            std::vector<float>& hOut) {
            const char* inName = ctx->getEngine().getIOTensorName(0);
            const char* outName = ctx->getEngine().getIOTensorName(1);
            ctx->setTensorAddress(inName, dInput);
            ctx->setTensorAddress(outName, dOutput);

            CUDA_CHECK(cudaMemcpyAsync(dInput, hIn.data(),
                                       hIn.size() * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start, stream));
            ctx->enqueueV3(stream);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            hOut.resize(volume(ctx->getEngine().getTensorShape(outName)));
            CUDA_CHECK(cudaMemcpyAsync(hOut.data(), dOutput,
                                       hOut.size() * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return ms;
        };

        // ------------------------------------------------------------
        // 4. Warmup + Timed Runs
        // ------------------------------------------------------------
        const int WARMUPS = 10;
        const int RUNS = 100;
        std::vector<std::vector<float>> chunkTimes(RUNS);
        std::vector<float> gpuTotals(RUNS), cpuTotals(RUNS);

        std::cout << "\n[INFO] Starting sequential inference ("
                  << WARMUPS << " warmups, " << RUNS << " runs)...\n\n";

        // Warmups
        for (int w = 0; w < WARMUPS; ++w) {
            nvtxRangePushA(("Warmup " + std::to_string(w + 1)).c_str());
            std::vector<float> hIn = hInput;
            for (int i = 0; i < numChunks; ++i) {
                std::vector<float> hOut;
                runChunk(contexts[i].get(), hIn, hOut);
                hIn = hOut;
            }
            nvtxRangePop();
        }

        // Timed runs
        for (int r = 0; r < RUNS; ++r) {
            nvtxRangePushA(("Inference " + std::to_string(r + 1)).c_str());
            auto t0 = std::chrono::high_resolution_clock::now();

            std::vector<float> hIn = hInput;
            std::vector<float> perChunk(numChunks);
            for (int i = 0; i < numChunks; ++i) {
                nvtxRangePushA(("Chunk " + std::to_string(i + 1)).c_str());
                std::vector<float> hOut;
                perChunk[i] = runChunk(contexts[i].get(), hIn, hOut);
                hIn = hOut;
                nvtxRangePop();
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            double cpuMs =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            double gpuMs = std::accumulate(perChunk.begin(), perChunk.end(), 0.0);

            chunkTimes[r] = perChunk;
            gpuTotals[r] = gpuMs;
            cpuTotals[r] = cpuMs;

            std::cout << "Run " << (r + 1)
                      << ": GPU " << gpuMs << " ms | CPU " << cpuMs << " ms\n";
            nvtxRangePop();
        }

        // ------------------------------------------------------------
        // 5. Save CSV
        // ------------------------------------------------------------
        std::ofstream f("chunk_timing.csv");
        f << "run_id,";
        for (int i = 0; i < numChunks; ++i)
            f << "chunk" << (i + 1) << "_gpu_ms"
              << (i + 1 < numChunks ? "," : "");
        f << ",total_gpu_ms,total_cpu_ms\n";

        for (int r = 0; r < RUNS; ++r) {
            f << (r + 1) << ",";
            for (int i = 0; i < numChunks; ++i)
                f << std::fixed << std::setprecision(4)
                  << chunkTimes[r][i] << (i + 1 < numChunks ? "," : "");
            f << "," << gpuTotals[r] << "," << cpuTotals[r] << "\n";
        }
        f.close();
        std::cout << "📝 Saved detailed timing to chunk_timing.csv\n";

        // ------------------------------------------------------------
        // 6. Cleanup
        // ------------------------------------------------------------
        cudaFree(dInput);
        cudaFree(dOutput);
        cudaStreamDestroy(stream);
        std::cout << "✅ Done.\n";
    } catch (std::exception const& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
