#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

// Simple logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
} gLogger;

struct Allocation {
    std::string name;
    void* address;
    size_t size;
    size_t offset_kb;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./profile_memory <path_to_engine>" << std::endl;
        return -1;
    }

    std::string enginePath = argv[1];
    
    // 1. Initialize Plugins (Crucial for TRT-LLM engines)
    initLibNvInferPlugins(&gLogger, "");

    // 2. Load Engine File
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "Error: Could not read engine file: " << enginePath << std::endl;
        return -1;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    // 3. Deserialize Engine
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size, nullptr);
    
    if (!engine) {
        std::cerr << "Error: Failed to deserialize engine. Ensure TRT-LLM libs are linked." << std::endl;
        return -1;
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    
    // 4. "Naive" Allocation Loop
    // We iterate through all IO tensors and allocate them using standard cudaMalloc
    std::vector<Allocation> allocs;
    std::vector<void*> bindings;
    
    std::cout << "\n--- Starting Naive Allocations (cudaMalloc) ---\n";

    int nbIOTensors = engine->getNbIOTensors();
    for (int i = 0; i < nbIOTensors; ++i) {
        const char* name = engine->getIOTensorName(i);
        nvinfer1::Dims dims = engine->getTensorShape(name);
        nvinfer1::DataType type = engine->getTensorDataType(name);

        // Calculate size (Simplified for demo: assuming batch 1, full dims)
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            // Handle dynamic dimensions by assuming max size (e.g., -1 becomes 512 or 1024)
            int dim_val = dims.d[d];
            if (dim_val == -1) dim_val = 1024; // Assuming max seq len
            vol *= dim_val;
        }

        // Bytes per element
        size_t elementSize = (type == nvinfer1::DataType::kFLOAT) ? 4 : 2; // Simplify: Float32 vs Float16
        size_t totalBytes = vol * elementSize;

        // Perform Allocation
        void* ptr = nullptr;
        cudaMalloc(&ptr, totalBytes);
        
        // Log it
        Allocation rec;
        rec.name = name;
        rec.address = ptr;
        rec.size = totalBytes;
        allocs.push_back(rec);
        
        context->setTensorAddress(name, ptr);
        
        std::cout << "Allocated: " << name << " | Size: " << totalBytes / 1024.0 << " KB" << std::endl;
    }

    // 5. Calculate Offsets relative to the first allocation
    // This simulates the "Virtual Memory Map" from your PDF
    if (!allocs.empty()) {
        // Sort by address to find the true order in memory
        std::sort(allocs.begin(), allocs.end(), [](const Allocation& a, const Allocation& b) {
            return a.address < b.address;
        });

        uintptr_t baseAddr = (uintptr_t)allocs[0].address;
        
        std::ofstream csv("memory_log.csv");
        csv << "Name,Offset_KB,Size_KB,Gap_Before_KB\n";

        for (size_t i = 0; i < allocs.size(); ++i) {
            uintptr_t currentAddr = (uintptr_t)allocs[i].address;
            size_t offset = currentAddr - baseAddr;
            
            // Calculate gap from previous block
            size_t gap = 0;
            if (i > 0) {
                uintptr_t prevEnd = (uintptr_t)allocs[i-1].address + allocs[i-1].size;
                gap = currentAddr - prevEnd;
            }

            csv << allocs[i].name << "," 
                << (offset / 1024.0) << "," 
                << (allocs[i].size / 1024.0) << ","
                << (gap / 1024.0) << "\n";
        }
        csv.close();
        std::cout << "\nMemory profile saved to 'memory_log.csv'. Run the python script to visualize.\n";
    }

    // Clean up
    // (Skipped for brevity, OS will reclaim)
    return 0;
}