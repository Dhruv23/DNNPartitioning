/*
Compile with:
nvcc -std=c++17 -O2 profile_memory.cpp -o profile_memory \
  -I/usr/local/cuda/include \
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

// ------------------------------------------------------------
// Memory Visualization Tools (Standard Allocator)
// ------------------------------------------------------------

struct AllocationRecord {
    std::string name;
    uintptr_t address;
    size_t size;
    std::string color;
};

// Global tracker for standard allocations
std::vector<AllocationRecord> g_allocations;

// Wrapper for cudaMalloc to track addresses
void cudaMallocTracked(void** devPtr, size_t size, const std::string& name) {
    CUDA_CHECK(cudaMalloc(devPtr, size));
    
    // Generate a color based on hash of name/index for consistency
    static int hue_offset = 0;
    std::string col = "hsl(" + std::to_string((hue_offset * 40) % 360) + ", 70%, 60%)";
    hue_offset++;

    g_allocations.push_back({name, reinterpret_cast<uintptr_t>(*devPtr), size, col});
}

void generateStandardHTMLVisualization(const std::string& filename) {
    if (g_allocations.empty()) return;

    // 1. Sort by address to establish the physical layout order
    std::sort(g_allocations.begin(), g_allocations.end(), 
              [](const AllocationRecord& a, const AllocationRecord& b) {
                  return a.address < b.address;
              });

    // 2. Determine address range
    uintptr_t base_addr = g_allocations.front().address;
    uintptr_t end_addr = g_allocations.back().address + g_allocations.back().size;
    size_t total_span = end_addr - base_addr;

    // 3. Visualization Constants (using 64KB blocks to match PagedAttention view)
    constexpr size_t VISUAL_BLOCK_SIZE = 64 * 1024; 
    size_t num_visual_blocks = (total_span + VISUAL_BLOCK_SIZE - 1) / VISUAL_BLOCK_SIZE;

    // Cap the blocks to prevent crashing browser if addresses are wildly far apart (e.g. > 2GB span)
    // If span is huge, we scale the visual block size.
    size_t actual_block_size = VISUAL_BLOCK_SIZE;
    if (num_visual_blocks > 20000) {
        actual_block_size = total_span / 5000; // Auto-scale to max 5000 blocks
    }

    std::ofstream html(filename);
    html << "<html><head><style>"
         << "body { font-family: 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; padding: 20px; display: flex; flex-direction: column; align-items: center; }"
         << "h1 { border-bottom: 1px solid #333; padding-bottom: 10px; width: 100%; max-width: 1200px; }"
         << ".container { width: 100%; max-width: 1200px; }"
         << ".dashboard { background: #1e1e1e; padding: 15px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }"
         << "table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }"
         << "th { text-align: left; padding: 8px; border-bottom: 2px solid #444; color: #aaa; }"
         << "td { padding: 8px; border-bottom: 1px solid #333; }"
         << ".color-box { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 8px; vertical-align: middle; }"
         << ".grid { display: flex; flex-wrap: wrap; gap: 2px; margin-top: 20px; }"
         << ".block { width: 12px; height: 12px; background: #333; border-radius: 2px; position: relative; }"
         << ".block:hover { border: 1px solid #fff; z-index: 10; transform: scale(1.5); }"
         << ".fragmentation { color: #f44336; font-weight: bold; }"
         << ".efficiency { color: #4caf50; font-weight: bold; }"
         << "</style></head><body>";

    html << "<div class='container'>";
    html << "<h1>Standard Allocator (cudaMalloc) Layout</h1>";

    // --- Stats Dashboard ---
    size_t total_alloc_bytes = 0;
    for(const auto& r : g_allocations) total_alloc_bytes += r.size;
    
    // Calculate gaps
    size_t total_gaps = 0;
    for (size_t i = 0; i < g_allocations.size() - 1; ++i) {
        uintptr_t current_end = g_allocations[i].address + g_allocations[i].size;
        uintptr_t next_start = g_allocations[i+1].address;
        if (next_start > current_end) {
            total_gaps += (next_start - current_end);
        }
    }

    // Calculate Percentages
    double util_pct = 0.0;
    double frag_pct = 0.0;
    
    if (total_span > 0) {
        util_pct = ((double)total_alloc_bytes / total_span) * 100.0;
        frag_pct = ((double)total_gaps / total_span) * 100.0;
    }

    html << "<div class='dashboard'>";
    html << "<h3>Allocation Report</h3>";
    html << "<table><thead><tr><th>Name</th><th>Size (MB)</th><th>Address Offset</th></tr></thead><tbody>";
    for(const auto& r : g_allocations) {
        html << "<tr>";
        html << "<td><span class='color-box' style='background:" << r.color << "'></span>" << r.name << "</td>";
        html << "<td>" << std::fixed << std::setprecision(3) << (r.size / 1024.0 / 1024.0) << "</td>";
        html << "<td>+" << (r.address - base_addr) / 1024 << " KB</td>";
        html << "</tr>";
    }
    html << "</tbody></table>";
    
    html << "<p style='margin-top:10px;'><b>Total Allocated:</b> " 
         << (total_alloc_bytes/1024.0/1024.0) << " MB (" 
         << "<span class='efficiency'>" << std::fixed << std::setprecision(1) << util_pct << "% Utilized</span>)</p>";
         
    html << "<p><b>External Fragmentation (Gaps):</b> <span class='fragmentation'>" 
         << (total_gaps/1024.0/1024.0) << " MB (" 
         << std::fixed << std::setprecision(1) << frag_pct << "% Unutilized)</span></p>";
    html << "</div>";

    // --- Grid Map ---
    html << "<h3>Virtual Memory Map (1 Block = " << (actual_block_size/1024.0) << " KB)</h3>";
    html << "<div class='grid'>";

    // We iterate through the specific allocated chunks and gaps to draw them
    // rather than iterating every byte to save time.
    uintptr_t current_ptr = base_addr;
    
    // Helper to draw N blocks of a certain color
    auto drawBlocks = [&](size_t bytes, std::string color, std::string title) {
        size_t blocks = (bytes + actual_block_size - 1) / actual_block_size;
        for(size_t i=0; i<blocks; ++i) {
             html << "<div class='block' style='background: " << color << "' title='" << title << "'></div>";
        }
    };

    for(const auto& rec : g_allocations) {
        // Draw Gap (if any) before this allocation
        if (rec.address > current_ptr) {
            size_t gap_size = rec.address - current_ptr;
            drawBlocks(gap_size, "#333", "Unused / Fragmentation");
        }
        
        // Draw Allocation
        drawBlocks(rec.size, rec.color, rec.name);
        current_ptr = rec.address + rec.size;
    }

    html << "</div></div></body></html>";
    html.close();
    std::cout << "📊 Generated Standard layout: " << filename << "\n";
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
        // Ensure "engines" directory exists
        if (fs::exists("engines") && fs::is_directory("engines")) {
             for (auto& e : fs::directory_iterator("engines"))
                if (e.path().extension() == ".engine")
                    enginePaths.push_back(e.path());
        }
       
        std::sort(enginePaths.begin(), enginePaths.end());
        if (enginePaths.empty())
            throw std::runtime_error("No .engine files found in 'engines/' directory.");

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
        // CHANGED: Use tracked malloc
        cudaMallocTracked(&dIn0, inElems * sizeof(float), "Network Input");

        std::vector<void*> dAct(numChunks);
        for (int i = 0; i < numChunks; ++i) {
            std::string name = "Chunk " + std::to_string(i+1) + " Output";
            // CHANGED: Use tracked malloc
            cudaMallocTracked(&dAct[i], volume(outDims[i]) * sizeof(float), name);
        }

        // GENERATE HTML VISUALIZATION
        generateStandardHTMLVisualization("memory_layout_LLM.html");

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