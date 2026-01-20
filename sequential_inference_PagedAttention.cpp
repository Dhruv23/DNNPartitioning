/*
nvcc -std=c++17 -O2 sequential_inference_PagedAttention.cpp -o run_trt_PagedAttention \
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
#include <sstream>
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
#include <sstream>

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

// ----------------------------------------------------------------------
// vLLM-Style Memory Management (Paper Section 4.2)
// ----------------------------------------------------------------------

// KEY CHANGE: Reduced Block Size to 64KB to minimize internal fragmentation.
// This achieves "High Utilization" while maintaining strict block boundaries.
constexpr size_t VLLM_BLOCK_SIZE = 64 * 1024; 

struct AllocationRecord {
    std::string name;
    size_t start_offset;
    size_t requested_size; // Actual data size
    size_t reserved_size;  // Aligned to block size
    std::string color;     // For visualization
};

class VLLMBlockAllocator {
public:
    explicit VLLMBlockAllocator(size_t total_bytes_needed) {
        // Calculate blocks needed
        size_t num_blocks = (total_bytes_needed + VLLM_BLOCK_SIZE - 1) / VLLM_BLOCK_SIZE;
        pool_size_ = num_blocks * VLLM_BLOCK_SIZE;

        std::cout << "\n[VLLM Allocator] Initializing Physical Memory Pool...\n";
        std::cout << "  Block Size: " << (VLLM_BLOCK_SIZE / 1024.0) << " KB\n";
        std::cout << "  Reserved:   " << (pool_size_ / 1024.0 / 1024.0) << " MB (" 
                  << num_blocks << " blocks)\n";

        CUDA_CHECK(cudaMalloc(&base_ptr_, pool_size_));
        current_offset_ = 0;
    }

    ~VLLMBlockAllocator() {
        if (base_ptr_) {
            cudaFree(base_ptr_);
            std::cout << "[VLLM Allocator] Physical Pool Freed.\n";
        }
    }

    void* allocate(size_t bytes, const std::string& tag = "Unknown") {
        if (bytes == 0) return nullptr;

        // Strict Block Alignment: We must take N whole blocks
        size_t blocks_needed = (bytes + VLLM_BLOCK_SIZE - 1) / VLLM_BLOCK_SIZE;
        size_t alloc_size = blocks_needed * VLLM_BLOCK_SIZE;

        if (current_offset_ + alloc_size > pool_size_) {
            throw std::runtime_error("VLLM OOM: Physical Block Pool Exhausted!");
        }

        void* ptr = static_cast<char*>(base_ptr_) + current_offset_;

        // Generate a random pastel color for visualization
        // Using distinct hues helps differentiate chunks visually
        static int hue_offset = 0;
        std::string col = "hsl(" + std::to_string((hue_offset * 40) % 360) + ", 70%, 60%)";
        hue_offset++;

        records_.push_back({tag, current_offset_, bytes, alloc_size, col});
        
        current_offset_ += alloc_size;
        return ptr;
    }

    // Generates a 'Block Map' HTML Visualization with Static Dashboard
    void generateHTMLVisualization(const std::string& filename) {
        std::ofstream html(filename);
        size_t total_blocks = pool_size_ / VLLM_BLOCK_SIZE;
        
        html << "<html><head><style>"
             << "body { font-family: 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; padding: 20px; display: flex; flex-direction: column; align-items: center; }"
             << "h1 { border-bottom: 1px solid #333; padding-bottom: 10px; width: 100%; max-width: 1200px; }"
             << ".container { width: 100%; max-width: 1200px; }"
             << ".dashboard { background: #1e1e1e; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }"
             << "table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }"
             << "th { text-align: left; padding: 8px; border-bottom: 2px solid #444; color: #aaa; }"
             << "td { padding: 8px; border-bottom: 1px solid #333; }"
             << ".util-high { color: #4caf50; font-weight: bold; }"
             << ".util-med { color: #ff9800; font-weight: bold; }"
             << ".color-box { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 8px; vertical-align: middle; }"
             << ".grid { display: flex; flex-wrap: wrap; gap: 2px; margin-top: 20px; }"
             << ".block { width: 12px; height: 12px; background: #333; border-radius: 2px; position: relative; }"
             << ".block:hover { border: 1px solid #fff; z-index: 10; transform: scale(1.5); }"
             << "</style></head><body>";

        html << "<div class='container'>";
        html << "<h1>PagedAttention Memory Analysis</h1>";
        
        // --- STATIC DASHBOARD (The "Slideshow" Section) ---
        html << "<div class='dashboard'>";
        html << "<h3 style='margin-top:0;'>Allocation Efficiency Report (Page Size: " << (VLLM_BLOCK_SIZE/1024.0) << " KB)</h3>";
        html << "<table>";
        html << "<thead><tr><th>Allocation Name</th><th>Size (MB)</th><th>Pages Used</th><th>Utilization</th><th>Fragmentation (KB)</th></tr></thead>";
        html << "<tbody>";
        
        size_t total_wasted_bytes = 0;
        size_t total_requested_bytes = 0;

        for (const auto& rec : records_) {
            double size_mb = rec.requested_size / 1024.0 / 1024.0;
            size_t pages = rec.reserved_size / VLLM_BLOCK_SIZE;
            double util = (double)rec.requested_size / rec.reserved_size * 100.0;
            double waste_kb = (rec.reserved_size - rec.requested_size) / 1024.0;
            
            total_requested_bytes += rec.requested_size;
            total_wasted_bytes += (rec.reserved_size - rec.requested_size);

            std::string util_class = (util > 95.0) ? "util-high" : "util-med";

            html << "<tr>";
            html << "<td><span class='color-box' style='background:" << rec.color << "'></span>" << rec.name << "</td>";
            html << "<td>" << std::fixed << std::setprecision(3) << size_mb << "</td>";
            html << "<td>" << pages << "</td>";
            html << "<td class='" << util_class << "'>" << std::setprecision(2) << util << "%</td>";
            html << "<td>" << std::setprecision(2) << waste_kb << " KB</td>";
            html << "</tr>";
        }
        html << "</tbody></table>";
        
        double global_eff = (double)total_requested_bytes / (total_requested_bytes + total_wasted_bytes) * 100.0;
        html << "<div style='margin-top:15px; text-align:right; font-size:16px;'>";
        html << "<b>Global Memory Efficiency: </b><span class='util-high'>" << std::fixed << std::setprecision(2) << global_eff << "%</span>";
        html << "</div></div>";

        // --- GRID VISUALIZATION ---
        html << "<h3>Physical Block Map (" << total_blocks << " Blocks)</h3>";
        html << "<div class='grid'>";

        for (size_t i = 0; i < total_blocks; ++i) {
            size_t block_start = i * VLLM_BLOCK_SIZE;
            
            // Identify owner
            const AllocationRecord* owner = nullptr;
            for (const auto& rec : records_) {
                if (block_start >= rec.start_offset && block_start < (rec.start_offset + rec.reserved_size)) {
                    owner = &rec;
                    break;
                }
            }

            if (owner) {
                // Visual style
                size_t alloc_end_data = owner->start_offset + owner->requested_size;
                size_t block_end = block_start + VLLM_BLOCK_SIZE;
                
                std::string style = "background: " + owner->color + ";";
                // If last block and significantly empty, make it slightly transparent
                if (alloc_end_data < block_end) {
                     style += "opacity: 0.8;"; 
                }

                html << "<div class='block' style='" << style << "'></div>";
            } else {
                html << "<div class='block' title='Free'></div>";
            }
        }
        html << "</div>"; // End Grid
        html << "</div></body></html>";
        
        html.close();
        std::cout << "📊 Generated PagedAttention map: " << filename << "\n";
    }

private:
    void* base_ptr_ = nullptr;
    size_t pool_size_ = 0;
    size_t current_offset_ = 0;
    std::vector<AllocationRecord> records_;
};


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
        // Shapes & Memory Planning
        // ------------------------------------------------------------
        const int kN = 1, kC = 3, kH = 224, kW = 224;
        std::vector<nvinfer1::Dims> inDims(numChunks), outDims(numChunks);
        
        // Calculate Total GPU Memory Needed with Block Alignment
        size_t total_gpu_mem_needed = 0;

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

            // Calculate aligned size for this chunk's output
            size_t raw_bytes = volume(outDims[i]) * sizeof(float);
            size_t aligned_bytes = (raw_bytes + VLLM_BLOCK_SIZE - 1) / VLLM_BLOCK_SIZE * VLLM_BLOCK_SIZE;
            total_gpu_mem_needed += aligned_bytes;
        }
        
        // Calculate aligned size for Input
        size_t inElems = volume(inDims[0]);
        size_t in_raw_bytes = inElems * sizeof(float);
        size_t in_aligned_bytes = (in_raw_bytes + VLLM_BLOCK_SIZE - 1) / VLLM_BLOCK_SIZE * VLLM_BLOCK_SIZE;
        total_gpu_mem_needed += in_aligned_bytes;

        auto finalOut = outDims.back();
        size_t outElems = volume(finalOut);

        // ------------------------------------------------------------
        // Buffers & Allocator
        // ------------------------------------------------------------
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::vector<float> hIn(inElems), hOut(outElems);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.f, 1.f);
        for (auto& v : hIn) v = dist(rng);

        // Instantiate Allocator with High-Precision (64KB) blocks
        VLLMBlockAllocator block_engine(total_gpu_mem_needed);

        void* dIn0;
        dIn0 = block_engine.allocate(inElems * sizeof(float), "Network Input");

        std::vector<void*> dAct(numChunks);
        for (int i = 0; i < numChunks; ++i) {
            size_t size = volume(outDims[i]) * sizeof(float);
            std::string label = "Chunk " + std::to_string(i+1) + " Out";
            dAct[i] = block_engine.allocate(size, label);
        }

        // Generate the visual report
        block_engine.generateHTMLVisualization("memory_layout_paged.html");

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

        cudaStreamDestroy(stream);
        std::cout << "✅ Done.\n";
    } catch (std::exception const& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
}