/*

nvcc -std=c++17 -O2 sequential_inference_graph.cpp -o run_trt_graph \
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

        // ------------------------------------------------------------
        // Load engines and contexts
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
        // Allocate buffers
        // ------------------------------------------------------------
        const int B = 1, C = 3, H = 224, W = 224;
        nvinfer1::Dims inputDims{4, {B, C, H, W}};
        std::vector<nvinfer1::Dims> inDims(numChunks), outDims(numChunks);

        for (int i = 0; i < numChunks; ++i) {
            const char* inName = engines[i]->getIOTensorName(0);
            const char* outName = engines[i]->getIOTensorName(1);
            inDims[i] = engines[i]->getTensorShape(inName);
            outDims[i] = engines[i]->getTensorShape(outName);
            if (std::any_of(inDims[i].d, inDims[i].d + inDims[i].nbDims,
                            [](int d) { return d < 0; }))
                inDims[i] = inputDims;
            contexts[i]->setInputShape(inName, inDims[i]);
        }

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* dIn0 = nullptr;
        CUDA_CHECK(cudaMalloc(&dIn0, volume(inDims[0]) * sizeof(float)));
        std::vector<void*> dAct(numChunks);
        for (int i = 0; i < numChunks; ++i)
            CUDA_CHECK(cudaMalloc(&dAct[i], volume(outDims[i]) * sizeof(float)));

        // Random input
        std::vector<float> hIn(volume(inDims[0]));
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.f, 1.f);
        for (auto& v : hIn) v = dist(rng);
        CUDA_CHECK(cudaMemcpyAsync(dIn0, hIn.data(),
                                   volume(inDims[0]) * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));

        std::cout << "[INFO] Allocated device buffers.\n";

        // ------------------------------------------------------------
        // Define lambda for enqueue
        // ------------------------------------------------------------
        auto enqueueChunk = [&](int i, void* dSrc, void* dDst) {
            const char* inName = engines[i]->getIOTensorName(0);
            const char* outName = engines[i]->getIOTensorName(1);
            contexts[i]->setTensorAddress(inName, dSrc);
            contexts[i]->setTensorAddress(outName, dDst);
            if (!contexts[i]->enqueueV3(stream))
                throw std::runtime_error("enqueueV3 failed");
        };

        // ------------------------------------------------------------
        // Warmups (outside graph)
        // ------------------------------------------------------------
        for (int w = 0; w < 3; ++w) {
            for (int i = 0; i < numChunks; ++i) {
                void* src = (i == 0) ? dIn0 : dAct[i - 1];
                enqueueChunk(i, src, dAct[i]);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // ------------------------------------------------------------
        // Capture CUDA graph of the full chain
        // ------------------------------------------------------------
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graphExec = nullptr;
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        {
            for (int i = 0; i < numChunks; ++i) {
                void* src = (i == 0) ? dIn0 : dAct[i - 1];
                enqueueChunk(i, src, dAct[i]);
            }
        }
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
        std::cout << "[INFO] Captured CUDA Graph with " << numChunks
                  << " enqueueV3 nodes.\n";

        // ------------------------------------------------------------
        // Timed runs using graph replay
        // ------------------------------------------------------------
        const int kRuns = 100;
        std::vector<double> gpuTotals(kRuns), cpuTotals(kRuns);

        for (int r = 0; r < kRuns; ++r) {
            nvtxRangePushA(("Run " + std::to_string(r + 1)).c_str());

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventRecord(start, stream));
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
            CUDA_CHECK(cudaEventRecord(stop, stream));
            CUDA_CHECK(cudaEventSynchronize(stop));
            auto t1 = std::chrono::high_resolution_clock::now();

            float gpuMs = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&gpuMs, start, stop));
            double cpuMs =
                std::chrono::duration<double, std::milli>(t1 - t0).count();

            gpuTotals[r] = gpuMs;
            cpuTotals[r] = cpuMs;

            std::cout << "Run " << (r + 1)
                      << ": GPU " << gpuMs << " ms | CPU " << cpuMs << " ms\n";

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            nvtxRangePop();
        }

        // ------------------------------------------------------------
        // Save results to chunk_timing.csv
        // ------------------------------------------------------------
        std::ofstream f("chunk_timing.csv");
        f << "run_id,total_gpu_ms,total_cpu_ms\n";
        for (int r = 0; r < kRuns; ++r)
            f << (r + 1) << ","
              << std::fixed << std::setprecision(6)
              << gpuTotals[r] << "," << cpuTotals[r] << "\n";
        std::cout << "📝 Saved timing results to chunk_timing.csv\n";

        double avgGpu = std::accumulate(gpuTotals.begin(), gpuTotals.end(), 0.0) /
                        gpuTotals.size();
        double avgCpu = std::accumulate(cpuTotals.begin(), cpuTotals.end(), 0.0) /
                        cpuTotals.size();
        std::cout << "\n✅ Average GPU: " << avgGpu
                  << " ms | Average CPU: " << avgCpu << " ms\n";

        // ------------------------------------------------------------
        // Cleanup
        // ------------------------------------------------------------
        CUDA_CHECK(cudaGraphDestroy(graph));
        CUDA_CHECK(cudaGraphExecDestroy(graphExec));
        for (auto& p : dAct) cudaFree(p);
        cudaFree(dIn0);
        cudaStreamDestroy(stream);
        std::cout << "✅ Done.\n";
    } catch (std::exception const& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
