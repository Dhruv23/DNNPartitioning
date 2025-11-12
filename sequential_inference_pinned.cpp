/*


nvcc -std=c++17 -O2 sequential_inference_gpu.cpp -o run_trt_pinned \
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

        // ============================================================
        // 1. Load all .engine chunks
        // ============================================================
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
            if (!e) throw std::runtime_error("Failed to deserialize: " + p.string());
            engines.emplace_back(e);
            contexts.emplace_back(e->createExecutionContext());
            std::cout << "[INFO] Loaded engine: " << p.filename().string() << "\n";
        }
        int numChunks = engines.size();
        std::cout << "✅ Loaded " << numChunks << " chunk engines.\n";

        // ============================================================
        // 2. Allocate pinned host + device buffers
        // ============================================================
        const int B = 1, C = 3, H = 224, W = 224;
        nvinfer1::Dims inputDims{4, {B, C, H, W}};

        // host input initialization
        std::vector<float> tmp(volume(inputDims));
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.f, 1.f);
        for (auto& v : tmp) v = dist(rng);

        // allocate pinned memory for inputs/outputs
        std::vector<void*> hIn(numChunks), hOut(numChunks);
        std::vector<nvinfer1::Dims> inDims(numChunks), outDims(numChunks);
        std::vector<void*> dAct(numChunks);
        void* dIn0 = nullptr;

        // create CUDA stream
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        for (int i = 0; i < numChunks; ++i) {
            auto* e = engines[i].get();
            const char* inName = e->getIOTensorName(0);
            const char* outName = e->getIOTensorName(1);

            inDims[i] = e->getTensorShape(inName);
            outDims[i] = e->getTensorShape(outName);
            if (std::any_of(inDims[i].d, inDims[i].d + inDims[i].nbDims,
                            [](int d) { return d < 0; }))
                inDims[i] = inputDims;  // set static input
            contexts[i]->setInputShape(inName, inDims[i]);

            CUDA_CHECK(cudaMalloc(&dAct[i], volume(outDims[i]) * sizeof(float)));
            CUDA_CHECK(cudaHostAlloc(&hIn[i],
                                     volume(inDims[i]) * sizeof(float),
                                     cudaHostAllocDefault));
            CUDA_CHECK(cudaHostAlloc(&hOut[i],
                                     volume(outDims[i]) * sizeof(float),
                                     cudaHostAllocDefault));
        }
        CUDA_CHECK(cudaMalloc(&dIn0, volume(inDims[0]) * sizeof(float)));

        std::memcpy(hIn[0], tmp.data(),
                    volume(inDims[0]) * sizeof(float));

        std::cout << "[INFO] Allocated pinned host + device buffers.\n";

        // ============================================================
        // 3. Helper lambda for one chunk
        // ============================================================
        auto runChunk = [&](int i, void* dSrc, void* dDst) {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));
            const char* inName = engines[i]->getIOTensorName(0);
            const char* outName = engines[i]->getIOTensorName(1);
            contexts[i]->setTensorAddress(inName, dSrc);
            contexts[i]->setTensorAddress(outName, dDst);
            CUDA_CHECK(cudaEventRecord(start, stream));
            contexts[i]->enqueueV3(stream);
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            return ms;
        };

        // ============================================================
        // 4. Warmup + timed runs
        // ============================================================
        int warmups = 5, runs = 20;
        std::cout << "\n[INFO] Starting sequential inference (" << warmups
                  << " warmups, " << runs << " runs)...\n\n";

        for (int w = 0; w < warmups; ++w) {
            nvtxRangePushA(("Warmup " + std::to_string(w + 1)).c_str());
            CUDA_CHECK(cudaMemcpyAsync(dIn0, hIn[0],
                                       volume(inDims[0]) * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
            for (int i = 0; i < numChunks; ++i) {
                void* src = (i == 0) ? dIn0 : dAct[i - 1];
                runChunk(i, src, dAct[i]);
                CUDA_CHECK(cudaMemcpyAsync(hOut[i], dAct[i],
                                           volume(outDims[i]) * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            nvtxRangePop();
        }

        std::vector<std::vector<float>> chunkTimes(runs);
        std::vector<float> gpuTotals(runs), cpuTotals(runs);

        for (int r = 0; r < runs; ++r) {
            nvtxRangePushA(("Run " + std::to_string(r + 1)).c_str());
            auto t0 = std::chrono::high_resolution_clock::now();

            CUDA_CHECK(cudaMemcpyAsync(dIn0, hIn[0],
                                       volume(inDims[0]) * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));

            std::vector<float> perChunk(numChunks);
            for (int i = 0; i < numChunks; ++i) {
                nvtxRangePushA(("Chunk " + std::to_string(i + 1)).c_str());
                void* src = (i == 0) ? dIn0 : dAct[i - 1];
                perChunk[i] = runChunk(i, src, dAct[i]);
                CUDA_CHECK(cudaMemcpyAsync(hOut[i], dAct[i],
                                           volume(outDims[i]) * sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
                nvtxRangePop();
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto t1 = std::chrono::high_resolution_clock::now();
            double cpuMs =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            double gpuMs = std::accumulate(perChunk.begin(), perChunk.end(), 0.0);
            chunkTimes[r] = perChunk;
            gpuTotals[r] = gpuMs;
            cpuTotals[r] = cpuMs;

            std::cout << "Run " << (r + 1) << ": GPU " << gpuMs
                      << " ms | CPU " << cpuMs << " ms\n";
            nvtxRangePop();
        }

        // ============================================================
        // 5. CSV export
        // ============================================================
        std::ofstream f("chunk_timing.csv");
        f << "run_id,";
        for (int i = 0; i < numChunks; ++i)
            f << "chunk" << (i + 1) << "_gpu_ms"
              << (i + 1 < numChunks ? "," : "");
        f << ",total_gpu_ms,total_cpu_ms\n";

        for (int r = 0; r < runs; ++r) {
            f << (r + 1) << ",";
            for (int i = 0; i < numChunks; ++i)
                f << std::fixed << std::setprecision(4)
                  << chunkTimes[r][i] << (i + 1 < numChunks ? "," : "");
            f << "," << gpuTotals[r] << "," << cpuTotals[r] << "\n";
        }
        std::cout << "📝 Saved detailed timing to chunk_timing.csv\n";

        // cleanup
        for (int i = 0; i < numChunks; ++i) {
            cudaFreeHost(hIn[i]);
            cudaFreeHost(hOut[i]);
            cudaFree(dAct[i]);
        }
        cudaFree(dIn0);
        cudaStreamDestroy(stream);
        std::cout << "✅ Done.\n";
    } catch (std::exception const& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
