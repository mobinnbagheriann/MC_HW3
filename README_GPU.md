# GPU Mode Setup and Usage Guide

This guide explains how to build and run the GPU-accelerated regex matching using cuDF/RAPIDS.

## Prerequisites

### 1. CUDA Installation
- NVIDIA GPU with compute capability 6.0 or higher
- CUDA Toolkit 11.0 or later
- NVIDIA Driver compatible with CUDA version

### 2. RAPIDS cuDF Installation

#### Option A: Using Conda (Recommended)
```bash
# Create a new conda environment
conda create -n rapids-env python=3.9

# Activate the environment
conda activate rapids-env

# Install RAPIDS cuDF
conda install -c rapidsai -c nvidia -c conda-forge cudf=23.10 python=3.9 cudatoolkit=11.8
```

#### Option B: Using Docker
```bash
# Pull RAPIDS container
docker pull rapidsai/rapidsai:23.10-cuda11.8-runtime-ubuntu22.04-py3.9

# Run container with your project mounted
docker run --gpus all -it -v /path/to/your/project:/workspace rapidsai/rapidsai:23.10-cuda11.8-runtime-ubuntu22.04-py3.9
```

### 3. Hyperscan Installation (for CPU mode)
```bash
sudo apt-get install libhyperscan-dev
```

## Project Structure

```
MC_HW_3/
├── src/
│   └── main.cu                 # Unified CPU/GPU source code
├── bin/                        # Build output directory
├── results/                    # Output files directory
├── Makefile                    # Unified build system
├── README_GPU.md              # This file
├── rules.txt                  # Regex patterns
├── set1.txt, set2.txt, set3.txt # Test datasets
└── *.tar.gz                   # Compressed datasets
```

## Building

### 1. Update Makefile paths
Edit `Makefile` and update these paths based on your installation:
```makefile
RAPIDS_ROOT := /opt/conda/envs/rapids-env  # or your conda env path
CUDA_ROOT := /usr/local/cuda               # or your CUDA installation path
```

### 2. Compile
```bash
make
```

If you encounter linking issues, you may need to set environment variables:
```bash
export LD_LIBRARY_PATH=/opt/conda/envs/rapids-env/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
make
```

## Usage

### CPU Mode (using Hyperscan)
```bash
./bin/HW3_MCC_030402_401110686 --mode cpu --rules rules.txt --input set1.txt --threads 4
```

### GPU Mode (using cuDF/RAPIDS)
```bash
./bin/HW3_MCC_030402_401110686 --mode gpu --rules rules.txt --input set1.txt
```

## Performance Testing

### Run CPU performance tests
```bash
make perf-test-cpu
```

### Run GPU performance test
```bash
make perf-test-gpu
```

### Run both CPU and GPU tests
```bash
make perf-test-all
```

### Other useful commands
```bash
make                    # Build the program
make run               # Quick demo with default settings
make clean             # Clean build artifacts
make help              # Show all available targets
```

## Output Files

The program generates two types of output files in the `results/` directory:

### 1. Match Results
- **CPU**: `Results_HW3_MCC_030402_401110686_CPU_{dataset}_{threads}.txt`
- **GPU**: `Results_HW3_MCC_030402_401110686_GPU_{dataset}_CUDA.txt`

Format: Each line contains comma-separated pattern IDs (0-indexed) that matched the input line.

### 2. Performance Metrics
- **CPU**: `Results_HW3_MCC_030402_401110686_CPU_{dataset}_Hyperscan.csv`
- **GPU**: `Results_HW3_MCC_030402_401110686_GPU_{dataset}_CUDA.csv`

**CPU CSV format:**
```
threads,throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency_ms
```

**GPU CSV format:**
```
matcher_name,throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency_ms
```

## Troubleshooting

### Common Issues

1. **cuDF headers not found**
   ```
   Solution: Update RAPIDS_ROOT path in Makefile
   ```

2. **CUDA runtime errors**
   ```
   Solution: Check NVIDIA driver and CUDA installation
   ```

3. **Library linking errors**
   ```
   Solution: Set LD_LIBRARY_PATH environment variable
   export LD_LIBRARY_PATH=/opt/conda/envs/rapids-env/lib:$LD_LIBRARY_PATH
   ```

4. **Hyperscan not found**
   ```
   Solution: Install libhyperscan-dev
   sudo apt-get install libhyperscan-dev
   ```

### Verification

To verify that GPU and CPU modes produce identical results:
```bash
# Run both modes
./bin/HW3_MCC_030402_401110686 --mode cpu --rules rules.txt --input set1.txt --threads 4
./bin/HW3_MCC_030402_401110686 --mode gpu --rules rules.txt --input set1.txt

# Compare results
diff results/Results_HW3_MCC_030402_401110686_CPU_set1_4.txt \
     results/Results_HW3_MCC_030402_401110686_GPU_set1_CUDA.txt
```

The files should be identical if both implementations are working correctly.

## Performance Notes

- GPU performance includes data transfer overhead (host-to-device and device-to-host)
- Performance may vary based on:
  - Number of patterns
  - Pattern complexity
  - Input data size
  - GPU memory bandwidth
  - CUDA compute capability

For optimal GPU performance, use larger datasets and complex regex patterns where GPU parallelization benefits outweigh data transfer overhead.
