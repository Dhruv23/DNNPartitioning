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

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
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
        auto runtime = std::shared_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(logger));

        // ------------------------------------------------------------
        // Load engines
        // ------------------------------------------------------------
        std::vector<fs::path> enginePaths;
        for (auto& e : fs::directory_iterator("engines"))
            if (e.path().extension() == ".engine")
                enginePaths.push_back(e.path());
        std::sort(enginePaths.begin(), enginePaths.end());
        if (enginePaths.empty())
            throw std::runtime_error("No .engine files found");

        std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> engines;
        std::vector<std::shared_ptr<nvinfer1::IExecutionContext>> contexts;

        for (auto& p : enginePaths) {
            auto bytes = loadFile(p);
            auto eng = std::shared_ptr<nvinfer1::ICudaEngine>(
                runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
            if (!eng) throw std::runtime_error("Failed to load engine: " + p.string());
            engines.push_back(eng);
            contexts.emplace_back(eng->createExecutionContext());
            std::cout << "[INFO] Loaded engine: " << p.filename() << "\n";
        }
        int numChunks = engines.size();
        std::cout << "✅ Loaded " << numChunks << " chunk engines.\n";

        // ------------------------------------------------------------
        // Shapes
        // ------------------------------------------------------------
        const int kN = 1, kC = 3, kH = 224, kW = 224;
        std::vector<nvinfer1::Dims> inDims(numChunks), outDims(numChunks);
        for (int i = 0; i < numChunks; ++i) {
            auto* e = engines[i].get();
            const char* inName = e->getIOTensorName(0);
            const char* outName = e->getIOTensorName(1);
            inDims[i] = e->getTensorShape(inName);
            outDims[i] = e->getTensorShape(outName);
            for (int d = 0; d < inDims[i].nbDims; ++d)
                if (inDims[i].d[d] == -1)
                    inDims[i] = nvinfer1::Dims4(kN, kC, kH, kW);
            contexts[i]->setInputShape(inName, inDims[i]);
        }
        auto finalOut = outDims.back();
        std::cout << "[INFO] Final output dims: [";
        for (int i = 0; i < finalOut.nbDims; ++i)
            std::cout << finalOut.d[i] << (i + 1 < finalOut.nbDims ? "," : "");
        std::cout << "]\n";

        // ------------------------------------------------------------
        // Buffers
        // ------------------------------------------------------------
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        size_t inElems = volume(inDims[0]);
        size_t outElems = volume(finalOut);
        std::vector<float> hIn(inElems), hOut(outElems);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.f, 1.f);
        for (auto& v : hIn) v = dist(rng);

        void* dIn0;
        CUDA_CHECK(cudaMalloc(&dIn0, inElems * sizeof(float)));

        std::vector<void*> dAct(numChunks);
        for (int i = 0; i < numChunks; ++i)
            CUDA_CHECK(cudaMalloc(&dAct[i], volume(outDims[i]) * sizeof(float)));

        // ------------------------------------------------------------
        // Run helper
        // ------------------------------------------------------------
        auto runChunk = [&](int i, void* dIn, void* dOut) {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreateWithFlags(&start, cudaEventDefault));
            CUDA_CHECK(cudaEventCreateWithFlags(&stop, cudaEventDefault));

            const char* inName = engines[i]->getIOTensorName(0);
            const char* outName = engines[i]->getIOTensorName(1);
            contexts[i]->setTensorAddress(inName, dIn);
            contexts[i]->setTensorAddress(outName, dOut);

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

        const int kWarmups = 10, kRuns = 100;
        std::cout << "\n[INFO] Starting (" << kWarmups << " warmups, "
                  << kRuns << " runs)...\n\n";

        // Warmups
        for (int w = 0; w < kWarmups; ++w) {
            CUDA_CHECK(cudaMemcpyAsync(dIn0, hIn.data(), inElems * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
            for (int i = 0; i < numChunks; ++i)
                runChunk(i, i == 0 ? dIn0 : dAct[i - 1], dAct[i]);
            CUDA_CHECK(cudaMemcpyAsync(hOut.data(), dAct.back(),
                                       outElems * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Timed runs
        std::vector<std::vector<float>> chunkTimes(kRuns);
        std::vector<float> gpuTotals(kRuns), cpuTotals(kRuns);

        for (int r = 0; r < kRuns; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemcpyAsync(dIn0, hIn.data(), inElems * sizeof(float),
                                       cudaMemcpyHostToDevice, stream));
            std::vector<float> c(numChunks);
            for (int i = 0; i < numChunks; ++i)
                c[i] = runChunk(i, i == 0 ? dIn0 : dAct[i - 1], dAct[i]);
            CUDA_CHECK(cudaMemcpyAsync(hOut.data(), dAct.back(),
                                       outElems * sizeof(float),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            double cpuMs =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            double gpuMs = std::accumulate(c.begin(), c.end(), 0.0);
            chunkTimes[r] = c;
            gpuTotals[r] = gpuMs;
            cpuTotals[r] = cpuMs;
            std::cout << "Run " << (r + 1)
                      << ": GPU " << gpuMs << " ms | CPU " << cpuMs << " ms\n";
        }

        // CSV
        std::ofstream f("chunk_timing.csv");
        f << "run_id,";
        for (int i = 0; i < numChunks; ++i)
            f << "chunk" << (i + 1) << "_gpu_ms"
              << (i + 1 < numChunks ? "," : "");
        f << ",total_gpu_ms,total_cpu_ms\n";
        for (int r = 0; r < kRuns; ++r) {
            f << (r + 1) << ",";
            for (int i = 0; i < numChunks; ++i)
                f << chunkTimes[r][i] << (i + 1 < numChunks ? "," : "");
            f << "," << gpuTotals[r] << "," << cpuTotals[r] << "\n";
        }
        std::cout << "📝 Saved detailed timing to chunk_timing.csv\n";

        for (auto& p : dAct) cudaFree(p);
        cudaFree(dIn0);
        cudaStreamDestroy(stream);
        std::cout << "✅ Done.\n";
    } catch (std::exception const& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
