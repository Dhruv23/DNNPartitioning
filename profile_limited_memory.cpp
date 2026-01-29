/*
profile_limited_memory.cpp

Compile:

nvcc -std=c++17 -O2 profile_limited_memory.cpp -o profile_limited_memory \
  -I/usr/include/x86_64-linux-gnu \
  -I/usr/local/cuda/include \
  -L/lib/x86_64-linux-gnu \
  -L/usr/local/cuda/lib64 \
  -lnvinfer \
  -lnvinfer_plugin \
  -lnvonnxparser \
  -lcudart

Run:
./profile_limited_memory

Outputs:
- vram_usage.csv
- token_latency.csv
- report_gpt2_limited.html
*/

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
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

static bool startsWith(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

// ------------------------------------------------------------
// Allocation tracking + page-ish visualization (approximation)
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

// ------------------------------------------------------------
// VRAM logging via nvidia-smi (1Hz)
// ------------------------------------------------------------

std::atomic<bool> keep_logging{true};

void logVRAMUsage(const std::string& logfile) {
    std::ofstream out(logfile);
    out << "time_s,used_MB,total_MB\n";

    auto start = std::chrono::steady_clock::now();

    while (keep_logging.load()) {
        FILE* pipe = popen(
            "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits",
            "r"
        );

        if (pipe) {
            int used = -1, total = -1;
            if (fscanf(pipe, "%d, %d", &used, &total) == 2 && used >= 0 && total > 0) {
                auto now = std::chrono::steady_clock::now();
                double t = std::chrono::duration<double>(now - start).count();
                out << t << "," << used << "," << total << "\n";
                out.flush();
            }
            pclose(pipe);
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// ------------------------------------------------------------
// CSV reading helpers
// ------------------------------------------------------------

struct VramSample { double t; int usedMB; int totalMB; };
struct TokenSample { int token; double ms; double tokPerSec; };

static std::vector<VramSample> readVramCSV(const std::string& path) {
    std::ifstream in(path);
    std::vector<VramSample> samples;
    if (!in) return samples;

    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string a,b,c;
        if (!std::getline(ss, a, ',')) continue;
        if (!std::getline(ss, b, ',')) continue;
        if (!std::getline(ss, c, ',')) continue;

        VramSample s{};
        s.t = std::stod(a);
        s.usedMB = std::stoi(b);
        s.totalMB = std::stoi(c);
        samples.push_back(s);
    }
    return samples;
}

static std::vector<TokenSample> readTokenCSV(const std::string& path) {
    std::ifstream in(path);
    std::vector<TokenSample> samples;
    if (!in) return samples;

    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string a,b,c;
        if (!std::getline(ss, a, ',')) continue;
        if (!std::getline(ss, b, ',')) continue;
        if (!std::getline(ss, c, ',')) continue;

        TokenSample s{};
        s.token = std::stoi(a);
        s.ms = std::stod(b);
        s.tokPerSec = std::stod(c);
        samples.push_back(s);
    }
    return samples;
}

// ------------------------------------------------------------
// HTML report (allocations + VRAM + token latency)
// ------------------------------------------------------------

static void generateHTMLReport(const std::string& filename,
                               const std::string& engineNameForDisplay,
                               const std::string& vramCSV,
                               const std::string& tokenCSV) {
    std::sort(g_allocations.begin(), g_allocations.end(),
              [](const AllocationRecord& a, const AllocationRecord& b) {
                  return a.address < b.address;
              });

    auto vram = readVramCSV(vramCSV);
    auto toks = readTokenCSV(tokenCSV);

    // Page approximation (left unchanged)
    constexpr size_t PAGE_SIZE = 64 * 1024;

    uintptr_t base_addr = g_allocations.empty() ? 0 : g_allocations.front().address;
    uintptr_t end_addr  = g_allocations.empty() ? 0 : (g_allocations.back().address + g_allocations.back().size);
    size_t total_span   = (end_addr > base_addr) ? (end_addr - base_addr) : 0;
    size_t total_pages_mapped = (total_span + PAGE_SIZE - 1) / PAGE_SIZE;

    std::ofstream html(filename);
    html << "<html><head><meta charset='utf-8'><style>"
         << "body { font-family: 'Segoe UI', sans-serif; background:#121212; color:#e0e0e0; padding:20px; }"
         << ".container { max-width: 1200px; margin:0 auto; }"
         << "h1,h2 { border-bottom:1px solid #333; padding-bottom:10px; }"
         << ".card { background:#1e1e1e; padding:20px; border-radius:10px; margin-bottom:20px;"
         << "box-shadow: 0 4px 6px rgba(0,0,0,0.3);} "
         << "table { width:100%; border-collapse:collapse; margin-top:10px; font-size:14px; }"
         << "th { text-align:left; padding:10px; border-bottom:2px solid #444; color:#aaa; }"
         << "td { padding:10px; border-bottom:1px solid #333; }"
         << ".bar-container { width:100%; background:#222; border-radius:6px; overflow:hidden; display:flex; align-items:flex-end; }"
         << ".bar-block { border-right:1px solid #111; }"
         << ".legend-box { display:inline-block; width:12px; height:12px; margin-right:6px; border-radius:2px; }"
         << ".stat-ok { color:#4caf50; font-weight:bold; }"
         << ".stat-bad { color:#f44336; font-weight:bold; }"
         << "code { background:#0f0f0f; padding:2px 6px; border-radius:6px; }"
         << "</style></head><body><div class='container'>";

    html << "<h1>TensorRT Memory + Runtime Report</h1>";
    html << "<p>Engine: <code>" << engineNameForDisplay << "</code></p>";

    // Allocation table
    html << "<div class='card'><h2>I/O Buffer Allocations (cudaMalloc tracked)</h2>";
    html << "<p style='color:#aaa;'>Note: This section tracks only explicit I/O buffers allocated by this program, "
            "not TensorRT internal workspace/scratch allocations.</p>";

    html << "<table><thead><tr>"
         << "<th>Tensor</th><th>Size (MB)</th><th>Approx. 64KB Pages</th><th>Approx. Fragmentation (KB)</th>"
         << "</tr></thead><tbody>";

    size_t total_bytes_used = 0, total_fragmentation = 0, total_pages_active = 0;

    for (const auto& r : g_allocations) {
        size_t pages_needed = (r.size + PAGE_SIZE - 1) / PAGE_SIZE;
        size_t allocated_space = pages_needed * PAGE_SIZE;
        size_t frag = allocated_space - r.size;

        total_bytes_used += r.size;
        total_fragmentation += frag;
        total_pages_active += pages_needed;

        html << "<tr>"
             << "<td><span class='legend-box' style='background:" << r.color << "'></span>"
             << r.name << "</td>"
             << "<td>" << std::fixed << std::setprecision(3) << (r.size / 1024.0 / 1024.0) << "</td>"
             << "<td>" << pages_needed << "</td>"
             << "<td>" << (frag / 1024.0) << "</td>"
             << "</tr>";
    }
    html << "</tbody></table>";

    html << "<p>Total I/O Bytes: <span class='stat-ok'>"
         << (total_bytes_used / 1024.0 / 1024.0) << " MB</span></p>";
    html << "<p>Total I/O Pages (approx, 64KB): <span class='stat-ok'>"
         << total_pages_active << "</span> (~"
         << (total_pages_active * PAGE_SIZE / 1024.0 / 1024.0) << " MB)</p>";
    html << "<p>Internal Fragmentation (approx): <span class='stat-bad'>"
         << (total_fragmentation / 1024.0) << " KB</span></p>";
    html << "</div>";

    // Address-space page map
    if (!g_allocations.empty() && total_pages_mapped > 0) {
        html << "<div class='card'><h2>Approximate Address-Space Page Map (64KB blocks)</h2>";
        html << "<p style='color:#aaa;'>This is a visualization of the virtual address range spanned by your "
                "I/O allocations, bucketed into 64KB blocks. It is an approximation, not a true GPU residency map.</p>";
        html << "<div class='bar-container' style='flex-wrap: wrap; height:auto; gap:1px; padding:6px;'>";

        for (size_t i = 0; i < total_pages_mapped; ++i) {
            uintptr_t page_start = base_addr + (i * PAGE_SIZE);
            uintptr_t page_end = page_start + PAGE_SIZE;

            std::string color = "#333";
            std::string title = "Gap / Unused";

            for (const auto& r : g_allocations) {
                uintptr_t r_end = r.address + r.size;
                if (r.address < page_end && r_end > page_start) {
                    color = r.color;
                    title = r.name;
                    break;
                }
            }

            html << "<div class='bar-block' style='width: 10px; height: 16px; background:" << color
                 << "; border-radius:2px;' title='Block " << i << ": " << title << "'></div>";
        }

        html << "</div></div>";
    }

    // VRAM usage timeline
    {
        html << "<div class='card'><h2>VRAM Used (sampled 1 Hz via nvidia-smi)</h2>";
        if (vram.empty()) {
            html << "<p style='color:#aaa;'>No VRAM samples found (missing vram_usage.csv?).</p>";
        } else {
            int totalMB = vram.front().totalMB;
            html << "<p>Total VRAM: <span class='stat-ok'>" << totalMB << " MB</span></p>";
            html << "<p style='color:#aaa;'>Each block = 1 second. Height/color = utilization.</p>";
            html << "<div class='bar-container' style='flex-wrap: wrap; height:auto; gap:1px; padding:6px;'>";

            for (size_t i = 0; i < vram.size(); ++i) {
                const auto& s = vram[i];
                double frac = (s.totalMB > 0) ? (double)s.usedMB / (double)s.totalMB : 0.0;
                int hue = static_cast<int>(120.0 * (1.0 - std::min(1.0, std::max(0.0, frac))));
                int h = 6 + static_cast<int>(50.0 * frac);

                html << "<div class='bar-block' style='width: 10px; height:" << h
                     << "px; background:hsl(" << hue << ",70%,55%); border-radius:2px;' title='"
                     << "t=" << std::fixed << std::setprecision(2) << s.t << "s, "
                     << s.usedMB << " / " << s.totalMB << " MB"
                     << "'></div>";
            }

            html << "</div>";
        }
        html << "</div>";
    }

    // Token latency timeline
    {
        html << "<div class='card'><h2>Token Latency / Throughput</h2>";
        if (toks.empty()) {
            html << "<p style='color:#aaa;'>No token samples found (missing token_latency.csv?).</p>";
        } else {
            double maxMs = 0.0;
            double minMs = 1e9;
            double avgMs = 0.0;

            for (const auto& s : toks) {
                maxMs = std::max(maxMs, s.ms);
                minMs = std::min(minMs, s.ms);
                avgMs += s.ms;
            }
            avgMs /= toks.size();
            double avgTokSec = (avgMs > 0.0) ? (1000.0 / avgMs) : 0.0;

            html << "<p>Latency (ms): min <span class='stat-ok'>" << std::fixed << std::setprecision(3) << minMs
                 << "</span>, avg <span class='stat-ok'>" << avgMs
                 << "</span>, max <span class='stat-bad'>" << maxMs
                 << "</span></p>";
            html << "<p>Avg throughput: <span class='stat-ok'>" << std::fixed << std::setprecision(2)
                 << avgTokSec << " tok/s</span></p>";
            html << "<p style='color:#aaa;'>Each block = 1 token. Height/color = latency.</p>";

            html << "<div class='bar-container' style='flex-wrap: wrap; height:auto; gap:1px; padding:6px;'>";
            for (const auto& s : toks) {
                double frac = (maxMs > 0.0) ? (s.ms / maxMs) : 0.0;
                frac = std::min(1.0, std::max(0.0, frac));
                int hue = static_cast<int>(120.0 * (1.0 - frac));
                int h = 6 + static_cast<int>(50.0 * frac);

                html << "<div class='bar-block' style='width: 10px; height:" << h
                     << "px; background:hsl(" << hue << ",70%,55%); border-radius:2px;' title='"
                     << "token " << s.token << ": " << std::fixed << std::setprecision(3) << s.ms
                     << " ms (" << std::fixed << std::setprecision(2) << s.tokPerSec << " tok/s)"
                     << "'></div>";
            }
            html << "</div>";
        }
        html << "</div>";
    }

    html << "<div class='card'><h2>Files</h2>"
         << "<ul style='color:#aaa;'>"
         << "<li><code>vram_usage.csv</code> — VRAM used/total sampled at 1 Hz</li>"
         << "<li><code>token_latency.csv</code> — per-token enqueue latency + tok/s</li>"
         << "<li><code>" << filename << "</code> — this report</li>"
         << "</ul></div>";

    html << "</div></body></html>";
    html.close();

    std::cout << "📊 Generated HTML Report: " << filename << "\n";
}

// ------------------------------------------------------------
// NEW: VRAM pressure allocation (Approach 1)
// ------------------------------------------------------------

static void* applyVRAMPressure(double keep_free_frac, size_t safety_bytes) {
    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));

    size_t targetFree = static_cast<size_t>(static_cast<double>(totalB) * keep_free_frac);
    size_t toAllocate = 0;

    if (freeB > targetFree) toAllocate = freeB - targetFree;

    // Leave some safety margin to reduce OOM risk due to fragmentation/overheads.
    if (toAllocate > safety_bytes) toAllocate -= safety_bytes;
    else toAllocate = 0;

    void* pressurePtr = nullptr;
    if (toAllocate > 0) {
        cudaError_t e = cudaMalloc(&pressurePtr, toAllocate);
        if (e == cudaSuccess && pressurePtr) {
            std::cout << "[PRESSURE] Allocated "
                      << (toAllocate / 1024.0 / 1024.0) << " MB to reduce free VRAM.\n";
        } else {
            std::cout << "[PRESSURE] Failed to allocate pressure buffer: "
                      << cudaGetErrorString(e) << "\n";
            pressurePtr = nullptr;
        }
    } else {
        std::cout << "[PRESSURE] No pressure allocation applied (freeB <= targetFree or safety margin too large).\n";
    }

    size_t freeAfter = 0, totalAfter = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeAfter, &totalAfter));
    std::cout << "[PRESSURE] VRAM after pressure: free="
              << (freeAfter / 1024.0 / 1024.0) << " MB / total="
              << (totalAfter / 1024.0 / 1024.0) << " MB\n";

    return pressurePtr;
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main() {
    void* pressurePtr = nullptr;

    try {
        Logger logger;
        auto runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));

        // 1) Load engine
        fs::path enginePath = "engines/gpt2/rank0.engine";
        if (!fs::exists(enginePath)) {
            throw std::runtime_error("Engine file not found: " + enginePath.string());
        }

        auto bytes = loadFile(enginePath);
        auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(bytes.data(), bytes.size())
        );
        if (!engine) throw std::runtime_error("Failed to deserialize engine");

        auto context = std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        if (!context) throw std::runtime_error("Failed to create execution context");

        std::cout << "✅ Loaded engine: " << enginePath.filename() << "\n";

        // Create stream
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Start VRAM logger thread
        const std::string vramCSV = "vram_usage.csv";
        const std::string tokenCSV = "token_latency.csv";
        keep_logging.store(true);
        std::thread vramThread(logVRAMUsage, vramCSV);

        // ------------------------------------------------------------
        // NEW: Apply VRAM pressure BEFORE allocating/binding I/O buffers
        // ------------------------------------------------------------
        // Leave 10% VRAM free, plus a 256MB safety margin.
        // If you OOM, increase keep_free_frac (e.g., 0.20) or safety_bytes.
        pressurePtr = applyVRAMPressure(/*keep_free_frac=*/0.03,
                                /*safety_bytes=*/32ULL * 1024 * 1024);



        // 2) Set shapes (matches your decode-with-past profile)
        auto makeDims = [](std::initializer_list<int> ds) {
            nvinfer1::Dims d{};
            d.nbDims = static_cast<int>(ds.size());
            int i = 0;
            for (int v : ds) d.d[i++] = v;
            return d;
        };

        const int B = 1;
        const int PAST = 511;        // past_sequence_length
        const int MASK = PAST + 1;   // attention_mask length
        const int NHEAD = 16;
        const int HDIM = 64;

        // 3) Allocate & bind all IO tensors
        int nbIO = engine->getNbIOTensors();
        std::cout << "[INFO] Engine has " << nbIO << " IO tensors.\n";

        // Write token latency CSV header
        {
            std::ofstream out(tokenCSV);
            out << "token,latency_ms,tokens_per_sec\n";
        }

        // Set input shapes for dynamic inputs
        for (int i = 0; i < nbIO; ++i) {
            const char* cname = engine->getIOTensorName(i);
            std::string name(cname);

            auto ioMode = engine->getTensorIOMode(cname);
            if (ioMode != nvinfer1::TensorIOMode::kINPUT) continue;

            nvinfer1::Dims cur = engine->getTensorShape(cname);
            bool hasDyn = false;
            for (int d = 0; d < cur.nbDims; ++d) if (cur.d[d] == -1) hasDyn = true;
            if (!hasDyn) continue;

            nvinfer1::Dims desired{};
            if (name == "input_ids") {
                desired = makeDims({B, 1});
            } else if (name == "attention_mask") {
                desired = makeDims({B, MASK});
            } else if (startsWith(name, "past_key_values.")) {
                desired = makeDims({B, NHEAD, PAST, HDIM});
            } else {
                desired = cur;
            }

            if (!context->setInputShape(cname, desired)) {
                throw std::runtime_error("Failed to setInputShape for " + name);
            }
        }

        // Allocate buffers based on concrete shapes
        for (int i = 0; i < nbIO; ++i) {
            const char* cname = engine->getIOTensorName(i);
            std::string name(cname);

            nvinfer1::Dims dims = context->getTensorShape(cname);
            nvinfer1::DataType type = engine->getTensorDataType(cname);

            size_t elemCount = volume(dims);
            size_t typeSize = 4;
            if (type == nvinfer1::DataType::kHALF) typeSize = 2;
            if (type == nvinfer1::DataType::kINT8) typeSize = 1;
            if (type == nvinfer1::DataType::kINT32) typeSize = 4;

            size_t totalBytes = elemCount * typeSize;

            void* devPtr = nullptr;
            cudaMallocTracked(&devPtr, totalBytes, name);

            if (!context->setTensorAddress(cname, devPtr)) {
                throw std::runtime_error("Failed to setTensorAddress for " + name);
            }

            std::cout << "Mapped " << name << " shape=[";
            for (int d = 0; d < dims.nbDims; ++d) {
                std::cout << dims.d[d] << (d + 1 < dims.nbDims ? "x" : "");
            }
            std::cout << "] -> " << (totalBytes / 1024.0) << " KB\n";
        }

        // 4) Initialize inputs (zero-fill)
        for (int i = 0; i < nbIO; ++i) {
            const char* cname = engine->getIOTensorName(i);
            auto mode = engine->getTensorIOMode(cname);
            if (mode != nvinfer1::TensorIOMode::kINPUT) continue;

            const void* cptr = context->getTensorAddress(cname);
            if (!cptr) continue;
            void* ptr = const_cast<void*>(cptr);

            nvinfer1::Dims dims = context->getTensorShape(cname);
            size_t elemCount = volume(dims);
            nvinfer1::DataType type = engine->getTensorDataType(cname);

            size_t typeSize = 4;
            if (type == nvinfer1::DataType::kHALF) typeSize = 2;
            if (type == nvinfer1::DataType::kINT8) typeSize = 1;
            if (type == nvinfer1::DataType::kINT32) typeSize = 4;

            CUDA_CHECK(cudaMemsetAsync(ptr, 0, elemCount * typeSize, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 5) Token loop: enqueue once per token, measure latency
        constexpr int NUM_TOKENS = 4096;

        std::ofstream tokenOut(tokenCSV, std::ios::app);
        tokenOut << std::fixed << std::setprecision(6);

        std::cout << "\n[RUN] Executing " << NUM_TOKENS << " token steps...\n";

        for (int token = 0; token < NUM_TOKENS; ++token) {
            auto t0 = std::chrono::high_resolution_clock::now();

            bool ok = context->enqueueV3(stream);
            if (!ok) {
                throw std::runtime_error("enqueueV3 failed at token " + std::to_string(token));
            }

            CUDA_CHECK(cudaStreamSynchronize(stream));

            auto t1 = std::chrono::high_resolution_clock::now();
            double dt_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            double tok_per_s = (dt_ms > 0.0) ? (1000.0 / dt_ms) : 0.0;

            tokenOut << token << "," << dt_ms << "," << tok_per_s << "\n";
            tokenOut.flush();

            std::cout << "Token " << token << ": "
                      << std::fixed << std::setprecision(3) << dt_ms << " ms  ("
                      << std::fixed << std::setprecision(2) << tok_per_s << " tok/s)\n";
        }

        // Stop VRAM logger
        keep_logging.store(false);
        vramThread.join();

        // Generate HTML report
        generateHTMLReport("report_gpt2_limited.html", "engines/gpt2/rank0.engine", vramCSV, tokenCSV);

        // Clean up device allocations
        for (const auto& r : g_allocations) {
            cudaFree(reinterpret_cast<void*>(r.address));
        }

        // Free pressure buffer LAST (kept during run)
        if (pressurePtr) {
            cudaFree(pressurePtr);
            pressurePtr = nullptr;
        }

        CUDA_CHECK(cudaStreamDestroy(stream));

        std::cout << "✅ Done.\n";
        return 0;

    } catch (std::exception const& e) {
        keep_logging.store(false);
        std::cerr << "Exception: " << e.what() << "\n";

        // Best-effort cleanup
        if (pressurePtr) cudaFree(pressurePtr);
        return 1;
    }
}
