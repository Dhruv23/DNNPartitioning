/*
nvcc -std=c++17 -O2 online.cpp -o online \
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
        // 1. Load all chunk engines for EfficientNet and ResNet
        // ------------------------------------------------------------
        std::vector<fs::path> efficientNetEnginePaths;
        std::vector<fs::path> resNetEnginePaths;

        // Load EfficientNet chunks
        for (auto& e : fs::directory_iterator("engines2"))
            if (e.path().string().find("efficientnet") != std::string::npos)
                efficientNetEnginePaths.push_back(e.path());

        // Load ResNet chunks
        for (auto& e : fs::directory_iterator("engines2"))
            if (e.path().string().find("resnet50") != std::string::npos)
                resNetEnginePaths.push_back(e.path());

        std::sort(efficientNetEnginePaths.begin(), efficientNetEnginePaths.end());
        std::sort(resNetEnginePaths.begin(), resNetEnginePaths.end());

        if (efficientNetEnginePaths.empty() || resNetEnginePaths.empty())
            throw std::runtime_error("No .engine files found in ./engines2");

        // Initialize EfficientNet engines and contexts
        std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> efficientNetEngines;
        std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> efficientNetContexts;
        for (auto& p : efficientNetEnginePaths) {
            auto bytes = loadFile(p);
            auto* e = runtime->deserializeCudaEngine(bytes.data(), bytes.size());
            if (!e)
                throw std::runtime_error("Failed to load engine: " + p.string());
            efficientNetEngines.emplace_back(e);
            efficientNetContexts.emplace_back(e->createExecutionContext());
            std::cout << "[INFO] Loaded EfficientNet engine: " << p.filename().string() << "\n";
        }

        // Initialize ResNet engines and contexts
        std::vector<std::unique_ptr<nvinfer1::ICudaEngine>> resNetEngines;
        std::vector<std::unique_ptr<nvinfer1::IExecutionContext>> resNetContexts;
        for (auto& p : resNetEnginePaths) {
            auto bytes = loadFile(p);
            auto* e = runtime->deserializeCudaEngine(bytes.data(), bytes.size());
            if (!e)
                throw std::runtime_error("Failed to load engine: " + p.string());
            resNetEngines.emplace_back(e);
            resNetContexts.emplace_back(e->createExecutionContext());
            std::cout << "[INFO] Loaded ResNet engine: " << p.filename().string() << "\n";
        }

        int numEfficientNetChunks = efficientNetEngines.size();
        int numResNetChunks = resNetEngines.size();
        std::cout << "✅ Loaded " << numEfficientNetChunks << " EfficientNet chunks and " << numResNetChunks << " ResNet chunks.\n";

        // ------------------------------------------------------------
        // 2. Determine buffer sizes and total GPU memory needed
        // ------------------------------------------------------------
        const int B = 1, C = 3, H = 224, W = 224;
        nvinfer1::Dims inputDims{4, {B, C, H, W}};
        std::vector<nvinfer1::Dims> efficientNetInDims(numEfficientNetChunks), efficientNetOutDims(numEfficientNetChunks);
        std::vector<nvinfer1::Dims> resNetInDims(numResNetChunks), resNetOutDims(numResNetChunks);

        size_t totalBytesEfficientNet = 0, totalBytesResNet = 0;
        std::vector<size_t> efficientNetOffsets(numEfficientNetChunks + 1);
        std::vector<size_t> resNetOffsets(numResNetChunks + 1);

        // EfficientNet buffer sizes
        for (int i = 0; i < numEfficientNetChunks; ++i) {
            const char* inName = efficientNetEngines[i]->getIOTensorName(0);
            const char* outName = efficientNetEngines[i]->getIOTensorName(1);
            efficientNetInDims[i] = efficientNetEngines[i]->getTensorShape(inName);
            efficientNetOutDims[i] = efficientNetEngines[i]->getTensorShape(outName);

            if (std::any_of(efficientNetInDims[i].d, efficientNetInDims[i].d + efficientNetInDims[i].nbDims,
                            [](int d) { return d < 0; }))
                efficientNetInDims[i] = inputDims;
            efficientNetContexts[i]->setInputShape(inName, efficientNetInDims[i]);

            efficientNetOffsets[i] = totalBytesEfficientNet;
            totalBytesEfficientNet += volume(efficientNetOutDims[i]) * sizeof(float);
        }
        efficientNetOffsets[numEfficientNetChunks] = totalBytesEfficientNet;

        // ResNet buffer sizes
        for (int i = 0; i < numResNetChunks; ++i) {
            const char* inName = resNetEngines[i]->getIOTensorName(0);
            const char* outName = resNetEngines[i]->getIOTensorName(1);
            resNetInDims[i] = resNetEngines[i]->getTensorShape(inName);
            resNetOutDims[i] = resNetEngines[i]->getTensorShape(outName);

            if (std::any_of(resNetInDims[i].d, resNetInDims[i].d + resNetInDims[i].nbDims,
                            [](int d) { return d < 0; }))
                resNetInDims[i] = inputDims;
            resNetContexts[i]->setInputShape(inName, resNetInDims[i]);

            resNetOffsets[i] = totalBytesResNet;
            totalBytesResNet += volume(resNetOutDims[i]) * sizeof(float);
        }
        resNetOffsets[numResNetChunks] = totalBytesResNet;

        size_t inputBytes = volume(inputDims) * sizeof(float);
        totalBytesEfficientNet += inputBytes;
        totalBytesResNet += inputBytes;

        std::cout << "[INFO] Total EfficientNet GPU memory = "
                  << (double)totalBytesEfficientNet / (1024 * 1024) << " MB\n";
        std::cout << "[INFO] Total ResNet GPU memory = "
                  << (double)totalBytesResNet / (1024 * 1024) << " MB\n";

        // ------------------------------------------------------------
        // 3. Allocate one big device buffer for each model
        // ------------------------------------------------------------
        void* dBaseEfficientNet = nullptr;
        void* dBaseResNet = nullptr;
        CUDA_CHECK(cudaMalloc(&dBaseEfficientNet, totalBytesEfficientNet));
        CUDA_CHECK(cudaMalloc(&dBaseResNet, totalBytesResNet));

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* dInput = dBaseEfficientNet;
        std::vector<void*> dEfficientNetOut(numEfficientNetChunks);
        size_t currentOffset = inputBytes;
        for (int i = 0; i < numEfficientNetChunks; ++i) {
            dEfficientNetOut[i] = static_cast<void*>(static_cast<char*>(dBaseEfficientNet) + currentOffset);
            currentOffset += volume(efficientNetOutDims[i]) * sizeof(float);
        }

        void* dResNetInput = dBaseResNet;
        std::vector<void*> dResNetOut(numResNetChunks);
        currentOffset = inputBytes;
        for (int i = 0; i < numResNetChunks; ++i) {
            dResNetOut[i] = static_cast<void*>(static_cast<char*>(dBaseResNet) + currentOffset);
            currentOffset += volume(resNetOutDims[i]) * sizeof(float);
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
        void* dEfficientNetSrc = dInput;
        for (int i = 0; i < numEfficientNetChunks; ++i) {
            const char* inName = efficientNetEngines[i]->getIOTensorName(0);
            const char* outName = efficientNetEngines[i]->getIOTensorName(1);
            efficientNetContexts[i]->setTensorAddress(inName, dEfficientNetSrc);
            efficientNetContexts[i]->setTensorAddress(outName, dEfficientNetOut[i]);
            dEfficientNetSrc = dEfficientNetOut[i];
        }

        void* dResNetSrc = dResNetInput;
        for (int i = 0; i < numResNetChunks; ++i) {
            const char* inName = resNetEngines[i]->getIOTensorName(0);
            const char* outName = resNetEngines[i]->getIOTensorName(1);
            resNetContexts[i]->setTensorAddress(inName, dResNetSrc);
            resNetContexts[i]->setTensorAddress(outName, dResNetOut[i]);
            dResNetSrc = dResNetOut[i];
        }

        // ------------------------------------------------------------
        // 6. Run sequentially for both models (100 iterations)
        // ------------------------------------------------------------
        const int WARMUPS = 10;
        const int RUNS = 100;
        std::vector<float> gpuEfficientNetTotals(RUNS, 0);
        std::vector<float> cpuEfficientNetTotals(RUNS, 0);
        std::vector<std::vector<float>> efficientNetChunkTimes(numEfficientNetChunks, std::vector<float>(RUNS, 0));

        std::vector<float> gpuResNetTotals(RUNS, 0);
        std::vector<float> cpuResNetTotals(RUNS, 0);
        std::vector<std::vector<float>> resNetChunkTimes(numResNetChunks, std::vector<float>(RUNS, 0));

        auto runPass = [&](int r, bool record) {
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            auto t0 = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaEventRecord(start, stream));

            // Run EfficientNet model chunk by chunk
            for (int i = 0; i < numEfficientNetChunks; ++i) {
                auto chunkStart = std::chrono::high_resolution_clock::now();
                efficientNetContexts[i]->enqueueV3(stream);
                auto chunkEnd = std::chrono::high_resolution_clock::now();
                float chunkTime = std::chrono::duration<float, std::milli>(chunkEnd - chunkStart).count();
                efficientNetChunkTimes[i][r] = chunkTime;
            }

            // Wait for EfficientNet to finish
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // Run ResNet model chunk by chunk
            for (int i = 0; i < numResNetChunks; ++i) {
                auto chunkStart = std::chrono::high_resolution_clock::now();
                resNetContexts[i]->enqueueV3(stream);
                auto chunkEnd = std::chrono::high_resolution_clock::now();
                float chunkTime = std::chrono::duration<float, std::milli>(chunkEnd - chunkStart).count();
                resNetChunkTimes[i][r] = chunkTime;
            }

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
                gpuEfficientNetTotals[r] = gpuMs;
                cpuEfficientNetTotals[r] = cpuMs;
                gpuResNetTotals[r] = gpuMs;
                cpuResNetTotals[r] = cpuMs;
            }
        };

        // Warmups
        for (int i = 0; i < WARMUPS; ++i) runPass(i, false);

        // Runs
        for (int r = 0; r < RUNS; ++r) runPass(r, true);

        // ------------------------------------------------------------
        // 7. Save timing results to CSV
        // ------------------------------------------------------------
        std::ofstream f("chunk_timing.csv");
        f << "run_id,";
        // EfficientNet chunk times
        for (int i = 0; i < numEfficientNetChunks; ++i) {
            f << "efficientnet_chunk" << i + 1 << "_gpu_ms,";
        }
        // ResNet chunk times
        for (int i = 0; i < numResNetChunks; ++i) {
            f << "resnet_chunk" << i + 1 << "_gpu_ms,";
        }
        f << "total_gpu_ms,total_cpu_ms\n";

        for (int r = 0; r < RUNS; ++r) {
            f << (r + 1) << ",";

            // EfficientNet chunk times
            float totalEfficientNetGPU = 0;
            for (int i = 0; i < numEfficientNetChunks; ++i) {
                f << efficientNetChunkTimes[i][r] << ",";
                totalEfficientNetGPU += efficientNetChunkTimes[i][r];
            }

            // ResNet chunk times
            float totalResNetGPU = 0;
            for (int i = 0; i < numResNetChunks; ++i) {
                f << resNetChunkTimes[i][r] << ",";
                totalResNetGPU += resNetChunkTimes[i][r];
            }

            f << totalEfficientNetGPU + totalResNetGPU << ","
              << cpuEfficientNetTotals[r] + cpuResNetTotals[r] << "\n";
        }
        f.close();
        std::cout << "📝 Saved results to chunk_timing.csv\n";

        // ------------------------------------------------------------
        // 8. Cleanup
        // ------------------------------------------------------------
        cudaFree(dBaseEfficientNet);
        cudaFree(dBaseResNet);
        cudaStreamDestroy(stream);
        std::cout << "✅ Done.\n";

    } catch (std::exception const& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}
