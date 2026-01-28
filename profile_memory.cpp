/*
Compile with (Copy and Paste this block):

# 1. Find the pip-installed TensorRT library path (matches trtllm-build)
TRT_LIBS_DIR=$(python3 -c "import site; print(site.getusersitepackages() + '/tensorrt_libs')")

# 2. Compile linking specifically against those libraries
nvcc -std=c++17 -O2 profile_memory.cpp -o profile_memory \
  -I/usr/local/cuda/include \
  -L/usr/local/cuda/lib64 \
  -L$TRT_LIBS_DIR \
  -lnvinfer -lnvonnxparser -lcudart -lnvinfer_plugin \
  -Wl,-rpath=$TRT_LIBS_DIR

# 3. Run
./profile_memory
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
// Memory Visualization Tools
// ------------------------------------------------------------

struct AllocationRecord {
    std::string name;
    uintptr_t address;
    size_t size;
    std::string color;
};

std::vector<AllocationRecord> g_allocations;

void cudaMallocTracked(void** devPtr, size_t size, const std::string& name) {
    CUDA_CHECK(cudaMalloc(devPtr, size));
    
    static int hue_offset = 0;
    std::string col = "hsl(" + std::to_string((hue_offset * 45) % 360) + ", 70%, 60%)";
    hue_offset++;

    g_allocations.push_back({name, reinterpret_cast<uintptr_t>(*devPtr), size, col});
}

void generatePagedHTMLVisualization(const std::string& filename) {
    if (g_allocations.empty()) return;

    // Sort by address
    std::sort(g_allocations.begin(), g_allocations.end(), 
              [](const AllocationRecord& a, const AllocationRecord& b) {
                  return a.address < b.address;
              });

    uintptr_t base_addr = g_allocations.front().address;
    uintptr_t end_addr = g_allocations.back().address + g_allocations.back().size;
    size_t total_span = end_addr - base_addr;
    
    // Page Constants
    constexpr size_t PAGE_SIZE = 64 * 1024; // 64KB pages
    size_t total_pages_mapped = (total_span + PAGE_SIZE - 1) / PAGE_SIZE;

    std::ofstream html(filename);
    html << "<html><head><style>"
         << "body { font-family: 'Segoe UI', sans-serif; background: #121212; color: #e0e0e0; padding: 20px; }"
         << ".container { max-width: 1200px; margin: 0 auto; }"
         << "h1, h2 { border-bottom: 1px solid #333; padding-bottom: 10px; }"
         << ".card { background: #1e1e1e; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }"
         << "table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 14px; }"
         << "th { text-align: left; padding: 10px; border-bottom: 2px solid #444; color: #888; }"
         << "td { padding: 10px; border-bottom: 1px solid #333; }"
         << ".bar-container { width: 100%; height: 24px; background: #333; border-radius: 4px; overflow: hidden; position: relative; display: flex; }"
         << ".bar-block { height: 100%; border-right: 1px solid #1e1e1e; }"
         << ".legend-box { display: inline-block; width: 12px; height: 12px; margin-right: 5px; border-radius: 2px; }"
         << ".stat-highlight { color: #4caf50; font-weight: bold; }"
         << ".stat-bad { color: #f44336; font-weight: bold; }"
         << "</style></head><body><div class='container'>";

    html << "<h1>Paged Memory Analysis (64KB Pages)</h1>";
    html << "<p>Profiling Engine: <code>gpt2/rank0.engine</code></p>";

    // --- Statistics ---
    html << "<div class='card'><h2>Allocation Report</h2>";
    html << "<table><thead><tr>"
         << "<th>Allocation Name</th>"
         << "<th>Size (MB)</th>"
         << "<th>Pages Active</th>"
         << "<th>Utilization</th>"
         << "<th>Fragmentation (KB)</th>"
         << "</tr></thead><tbody>";

    size_t total_bytes_used = 0;
    size_t total_fragmentation = 0;
    size_t total_pages_active = 0;

    for(const auto& r : g_allocations) {
        size_t pages_needed = (r.size + PAGE_SIZE - 1) / PAGE_SIZE;
        size_t allocated_space = pages_needed * PAGE_SIZE;
        size_t frag = allocated_space - r.size;
        double util = (double)r.size / allocated_space * 100.0;

        total_bytes_used += r.size;
        total_fragmentation += frag;
        total_pages_active += pages_needed;

        html << "<tr>"
             << "<td><span class='legend-box' style='background:" << r.color << "'></span>" << r.name << "</td>"
             << "<td>" << std::fixed << std::setprecision(3) << (r.size / 1024.0 / 1024.0) << "</td>"
             << "<td>" << pages_needed << "</td>"
             << "<td>" << std::fixed << std::setprecision(2) << util << "%</td>"
             << "<td>" << (frag / 1024.0) << " KB</td>"
             << "</tr>";
    }
    html << "</tbody></table></div>";

    // --- Summary ---
    html << "<div class='card'><h2>Summary</h2>"
         << "<p>Total Data Size: <span class='stat-highlight'>" << (total_bytes_used / 1024.0 / 1024.0) << " MB</span></p>"
         << "<p>Total Pages Active: <span class='stat-highlight'>" << total_pages_active << "</span> (" << (total_pages_active * PAGE_SIZE / 1024.0 / 1024.0) << " MB physical)</p>"
         << "<p>Internal Fragmentation: <span class='stat-bad'>" << (total_fragmentation / 1024.0) << " KB</span> (wasted space inside active pages)</p>"
         << "</div>";

    // --- Visualization ---
    html << "<div class='card'><h2>Physical Page Map (1 Block = 64KB)</h2>";
    html << "<div class='bar-container' style='flex-wrap: wrap; height: auto; gap: 1px;'>";

    // Reconstruct the visual map
    // We iterate 64KB chunks from base_addr to end_addr
    for(size_t i = 0; i < total_pages_mapped; ++i) {
        uintptr_t page_start = base_addr + (i * PAGE_SIZE);
        uintptr_t page_end = page_start + PAGE_SIZE;
        
        std::string color = "#333"; // Default empty/gap
        std::string title = "Gap / Unused";

        // Check which allocation "owns" this page
        for(const auto& r : g_allocations) {
            uintptr_t r_end = r.address + r.size;
            // Simple overlap check
            if (r.address < page_end && r_end > page_start) {
                color = r.color;
                title = r.name;
                break;
            }
        }
        
        html << "<div class='bar-block' style='width: 12px; height: 20px; background: " << color 
             << ";' title='Page " << i << ": " << title << "'></div>";
    }

    html << "</div></div></div></body></html>";
    html.close();
    std::cout << "📊 Generated Paged Report: " << filename << "\n";
}

int main() {
    try {
        Logger logger;
        auto runtime = std::shared_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(logger));

        // ------------------------------------------------------------
        // 1. Explicitly Load GPT-2 Engine
        // ------------------------------------------------------------
        fs::path enginePath = "engines/gpt2/rank0.engine";
        if (!fs::exists(enginePath)) {
            throw std::runtime_error("Engine file not found: " + enginePath.string());
        }

        auto bytes = loadFile(enginePath);
        auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        
        if (!engine) throw std::runtime_error("Failed to deserialize engine");
        
        auto context = std::shared_ptr<nvinfer1::IExecutionContext>(
            engine->createExecutionContext());
            
        std::cout << "✅ Loaded engine: " << enginePath.filename() << "\n";

        // ------------------------------------------------------------
        // 2. Iterate ALL IO Tensors & Alloc
        // ------------------------------------------------------------
        int nbBindings = engine->getNbIOTensors();
        std::cout << "[INFO] Engine has " << nbBindings << " IO bindings.\n";

        for (int i = 0; i < nbBindings; ++i) {
            const char* name = engine->getIOTensorName(i);
            nvinfer1::Dims dims = engine->getTensorShape(name);
            nvinfer1::DataType type = engine->getTensorDataType(name);
            
            // Resolve dynamic shapes (-1) to fixed values for profiling
            nvinfer1::Dims resolvedDims = dims;
            for(int d=0; d<dims.nbDims; ++d) {
                if(resolvedDims.d[d] == -1) resolvedDims.d[d] = 512; // Assume 512 sequence len
                // If batch dim is first and -1, assume 1
                if(d == 0 && dims.d[d] == -1) resolvedDims.d[d] = 1;
            }

            // Calculate size
            size_t elemCount = volume(resolvedDims);
            size_t typeSize = 4; // Default float32
            if (type == nvinfer1::DataType::kHALF) typeSize = 2;
            if (type == nvinfer1::DataType::kINT8) typeSize = 1;
            if (type == nvinfer1::DataType::kINT32) typeSize = 4;
            
            size_t totalBytes = elemCount * typeSize;
            
            void* devPtr;
            cudaMallocTracked(&devPtr, totalBytes, std::string(name));
            context->setTensorAddress(name, devPtr);
            
            std::cout << "Mapped " << name << " -> " 
                      << (totalBytes / 1024.0) << " KB (" 
                      << (totalBytes + 65535)/65536 << " pages)\n";
        }

        // ------------------------------------------------------------
        // 3. Generate Report
        // ------------------------------------------------------------
        generatePagedHTMLVisualization("memory_layout_gpt2.html");
        
        // Clean up
        for(const auto& r : g_allocations) {
            cudaFree((void*)r.address);
        }

    } catch (std::exception const& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}