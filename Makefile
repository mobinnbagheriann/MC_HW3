# Makefile for HW3 – High-Performance Multithreaded Regex Matching
#
# Build targets:
#   make              - Optimized build (default)
#   make debug        - Debug build with symbols
#   make clean        - Remove binaries & result files content
#   make run          - Quick demo run (default parameters)
#   make perf-test-cpu- Comprehensive CPU performance sweep
#   make perf-test-gpu- Comprehensive GPU performance tests
#   make help         - Print this help message

# ---------------------------------------------------------------------------
# Toolchain
# Use NVCC as the primary compiler since we now have CUDA code
NVCC       := nvcc
NVCCFLAGS  := -O3 -std=c++17 -arch=sm_60 -Xcompiler -Wall,-Wextra,-fPIC

# Include paths for RAPIDS/cuDF (adjust these paths based on your installation)
RAPIDS_ROOT := /opt/conda/envs/rapids
CUDA_ROOT := /usr/local/cuda

INCLUDES := -I$(RAPIDS_ROOT)/include \
           -I$(CUDA_ROOT)/include \
           -I$(RAPIDS_ROOT)/include/libcudf/libcudacxx

# Library paths and linking
LDFLAGS := -L$(RAPIDS_ROOT)/lib \
          -L$(CUDA_ROOT)/lib64 \
          -L/usr/lib/x86_64-linux-gnu \
          -lcudf -lrmm -lcudart -lhs -lpthread

# ---------------------------------------------------------------------------
# Project layout
# NOTE: Your source code (main.cu) should be placed inside the 'src' directory.
SRC_DIR      := src
BIN_DIR      := bin
RESULTS_DIR  := results

# Using the format from the assignment PDF
TARGET := $(BIN_DIR)/HW3_MCC_030402_401106039
SRC    := $(SRC_DIR)/main.cu

# ---------------------------------------------------------------------------
# Default example parameters (handy for "make run")
# NOTE: Create these placeholder files for a quick test.
DEFAULT_RULES := rules.txt
DEFAULT_INPUT := set1.txt
DEFAULT_THREADS := 4
DEFAULT_MODE := cpu

# ---------------------------------------------------------------------------
# Build rules
.PHONY: all debug clean run perf-test-cpu perf-test-gpu perf-test-all help

all: $(TARGET)

debug: NVCCFLAGS := -g -O0 -std=c++17 -arch=sm_60 -Xcompiler -Wall,-Wextra,-fPIC
debug: INCLUDES := -I$(RAPIDS_ROOT)/include -I$(CUDA_ROOT)/include -I$(RAPIDS_ROOT)/include/libcudf/libcudacxx
debug: LDFLAGS := -L$(RAPIDS_ROOT)/lib -L$(CUDA_ROOT)/lib64 -L/usr/lib/x86_64-linux-gnu -lcudf -lrmm -lcudart -lhs -lpthread
debug: clean all

$(TARGET): $(SRC) | $(BIN_DIR)
	@echo "Compiling unified CPU/GPU regex matcher with cuDF/RAPIDS..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)
	@echo "Build completed successfully!"

$(BIN_DIR):
	mkdir -p $@

# The 'run' and 'perf-test' targets will create this directory
$(RESULTS_DIR):
	mkdir -p $@

clean:
	@echo "Cleaning up build artifacts and results..."
	@if [ -d "$(BIN_DIR)" ]; then \
		echo "Cleaning bin directory contents..."; \
		rm -f $(BIN_DIR)/*; \
	fi
	@if [ -d "$(RESULTS_DIR)" ]; then \
		echo "Cleaning results directory contents..."; \
		rm -f $(RESULTS_DIR)/*; \
	fi

# ---------------------------------------------------------------------------
# Quick functional sanity check
run: $(TARGET) | $(RESULTS_DIR)
	@echo "Running default demo with $(DEFAULT_MODE) mode and $(DEFAULT_THREADS) threads..."
	@echo "Rules: $(DEFAULT_RULES), Input: $(DEFAULT_INPUT)"
	@echo "Output files will be automatically generated in $(RESULTS_DIR)/"
	@$(TARGET) --mode $(DEFAULT_MODE) --rules $(DEFAULT_RULES) --input $(DEFAULT_INPUT) --threads $(DEFAULT_THREADS)
	@echo "Demo run complete."

# ---------------------------------------------------------------------------
# Comprehensive CPU performance sweep
# NOTE: This assumes you have input files named 'set1.txt', 'set2.txt', etc.
perf-test-cpu: $(TARGET)
	@echo "Running comprehensive CPU performance tests..."
	@mkdir -p $(RESULTS_DIR)
	@# Loop over different datasets
	@for dataset in set1 set2 set3; do \
		echo "================================================="; \
		echo "              Testing Dataset: $${dataset}.txt         "; \
		echo "================================================="; \
		for threads in 1 2 4 8 16 32; do \
			echo "Testing with $$threads thread(s)..."; \
			$(TARGET) --mode cpu --rules $(DEFAULT_RULES) --input $${dataset}.txt --threads $$threads; \
		done; \
	done
	@echo "CPU performance tests complete."

# ---------------------------------------------------------------------------
# Comprehensive GPU performance test
perf-test-gpu: $(TARGET)
	@echo "Running comprehensive GPU performance tests..."
	@mkdir -p $(RESULTS_DIR)
	@# Loop over different datasets
	@for dataset in set1.txt set2.txt set3.txt; do \
		echo "================================================="; \
		echo "              Testing Dataset: $${dataset}         "; \
		echo "================================================="; \
		echo "Testing with GPU..."; \
		$(TARGET) --mode gpu --rules $(DEFAULT_RULES) --input $$dataset; \
	done
	@echo "GPU performance tests complete."

# ---------------------------------------------------------------------------
# Run both CPU and GPU performance tests
perf-test-all: perf-test-cpu perf-test-gpu
	@echo "All performance tests completed!"

# ---------------------------------------------------------------------------
# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build the program with optimizations (default)"
	@echo "  debug            - Build with debug symbols and no optimization"
	@echo "  clean            - Remove build artifacts and result files"
	@echo "  run              - Run a quick demo with default settings"
	@echo "  perf-test-cpu    - Run a performance sweep for the CPU version"
	@echo "  perf-test-gpu    - Run GPU performance tests with cuDF/RAPIDS"
	@echo "  perf-test-all    - Run both CPU and GPU performance tests"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make                                    # Build the program"
	@echo "  make run                                # Quick demo"
	@echo "  make perf-test-cpu                      # CPU performance tests"
	@echo "  make perf-test-gpu                      # GPU performance tests"
	@echo "  make perf-test-all                      # Both CPU and GPU tests"
	@echo "  ./bin/HW3_MCC_030402_401106039 --mode cpu --rules rules.txt --input set1.txt --threads 4"
	@echo "  ./bin/HW3_MCC_030402_401106039 --mode gpu --rules rules.txt --input set1.txt"

