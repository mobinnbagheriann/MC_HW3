# Makefile for HW3 – High-Performance Multithreaded Regex Matching
#
# Build targets:
#   make              - Optimized build (default)
#   make debug        - Debug build with symbols
#   make clean        - Remove binaries & result files content
#   make run          - Quick demo run (default parameters)
#   make perf-test-cpu- Comprehensive CPU performance sweep
#   make perf-test-gpu- Comprehensive GPU performance tests
#   make perf-test-all- Run both CPU and GPU tests
#   make help         - Print this help message

# ---------------------------------------------------------------------------
# Toolchain
NVCC       := nvcc
NVCCFLAGS  := -O3 -std=c++17 -arch=sm_75 -Xcompiler -Wall,-Wextra,-fPIC

# ---------------------------------------------------------------------------
# RAPIDS/cuDF (from active conda env) + CUDA (from nvcc path)
RAPIDS_ROOT ?= $(CONDA_PREFIX)
ifeq ($(strip $(RAPIDS_ROOT)),)
  $(error No conda environment active. Run: 'conda activate rapids-25.08')
endif
CUDA_ROOT ?= $(shell dirname $$(dirname $$(realpath $$(which nvcc))))

# ---------------------------------------------------------------------------
# Hyperscan (no pkg-config): fallback order -> $(CONDA_PREFIX), /usr/local, /usr/local/hyperscan
HS_INC_DIRS := \
  $(RAPIDS_ROOT)/include \
  /usr/local/include \
  /usr/local/hyperscan/include
HS_LIB_DIRS := \
  $(RAPIDS_ROOT)/lib \
  /usr/local/lib \
  /usr/local/hyperscan/build/lib

# Compose include flags (keep both include and include/hs just in case)
HS_INC := $(foreach d,$(HS_INC_DIRS),-I$(d)) $(foreach d,$(HS_INC_DIRS),-I$(d)/hs)
HS_LIB := $(foreach d,$(HS_LIB_DIRS),-L$(d)) -lhs

# ---------------------------------------------------------------------------
# Includes / Libs
INCLUDES := -I$(RAPIDS_ROOT)/include \
            -I$(RAPIDS_ROOT)/include/libcudf/libcudacxx \
            -I$(CUDA_ROOT)/include \
            $(HS_INC)

LDFLAGS  := -L$(RAPIDS_ROOT)/lib \
            -L$(CUDA_ROOT)/lib64 \
            -L/usr/lib/x86_64-linux-gnu \
            $(HS_LIB) \
            -lcudf -lrmm -lcudart -lpthread -ldl \
            -Wl,-rpath,$(RAPIDS_ROOT)/lib \
            -Wl,-rpath,$(CUDA_ROOT)/lib64 \
            -Wl,-rpath,/usr/local/lib \
            -Wl,-rpath,/usr/local/hyperscan/build/lib

# ---------------------------------------------------------------------------
# Project layout (kept)
SRC_DIR      := src
BIN_DIR      := bin
RESULTS_DIR  := results

# Final binary name (kept)
TARGET := $(BIN_DIR)/HW3_MCC_030402_401106039

# Sources: main + any additional .cu/.cpp in repo (kept general)
CU_SRCS   := $(wildcard $(SRC_DIR)/*.cu) $(wildcard *.cu)
CPP_SRCS  := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard *.cpp)
OBJS      := $(CU_SRCS:.cu=.o) $(CPP_SRCS:.cpp=.o)

# ---------------------------------------------------------------------------
# Defaults for 'make run' (kept)
DEFAULT_RULES   := rules.txt
DEFAULT_INPUT   := set1.txt
DEFAULT_THREADS := 4
DEFAULT_MODE    := cpu

# ---------------------------------------------------------------------------
# Build rules
.PHONY: all debug clean run perf-test-cpu perf-test-gpu perf-test-all help

all: $(TARGET)

debug: NVCCFLAGS := -g -G -O0 -std=c++17 -arch=sm_75 -Xcompiler -Wall,-Wextra,-fPIC
debug: clean $(TARGET)

# compile .cu and .cpp to .o (in-place to keep your structure unchanged)
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -x c++ -c $< -o $@

$(TARGET): $(OBJS) | $(BIN_DIR)
	@echo "Linking $(TARGET) ..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(OBJS) -o $@ $(LDFLAGS)
	@echo "Build completed: $@"

$(BIN_DIR):
	mkdir -p $@

$(RESULTS_DIR):
	mkdir -p $@

clean:
	@echo "Cleaning up..."
	@rm -f $(OBJS) $(TARGET)
	@rm -f $(RESULTS_DIR)/*

# ---------------------------------------------------------------------------
# Quick demo run (kept)
run: $(TARGET) | $(RESULTS_DIR)
	@echo "Running default demo with $(DEFAULT_MODE) mode and $(DEFAULT_THREADS) threads..."
	@$(TARGET) --mode $(DEFAULT_MODE) --rules $(DEFAULT_RULES) --input $(DEFAULT_INPUT) --threads $(DEFAULT_THREADS)
	@echo "Demo run complete."

# ---------------------------------------------------------------------------
# CPU perf sweep (kept)
perf-test-cpu: $(TARGET) | $(RESULTS_DIR)
	@echo "Running comprehensive CPU performance tests..."
	@for dataset in set1 set2 set3; do \
		echo "================================================="; \
		echo "              Testing Dataset: $${dataset}.txt   "; \
		echo "================================================="; \
		for threads in 1 2 4 8 16 32; do \
			echo "Testing with $$threads thread(s)..."; \
			$(TARGET) --mode cpu --rules $(DEFAULT_RULES) --input $${dataset}.txt --threads $$threads; \
		done; \
	done
	@echo "CPU performance tests complete."

# ---------------------------------------------------------------------------
# GPU perf sweep (kept)
perf-test-gpu: $(TARGET) | $(RESULTS_DIR)
	@echo "Running comprehensive GPU performance tests..."
	@for dataset in set1.txt set2.txt set3.txt; do \
		echo "================================================="; \
		echo "              Testing Dataset: $${dataset}       "; \
		echo "================================================="; \
		echo "Testing with GPU..."; \
		$(TARGET) --mode gpu --rules $(DEFAULT_RULES) --input $${dataset}; \
	done
	@echo "GPU performance tests complete."

perf-test-all: perf-test-cpu perf-test-gpu

# ---------------------------------------------------------------------------
help:
	@echo "Targets:"
	@echo "  make              - Build optimized binary"
	@echo "  debug             - Build debug binary"
	@echo "  clean             - Remove generated files"
	@echo "  run               - Run a quick demo with default settings"
	@echo "  perf-test-cpu     - Run a performance sweep for the CPU version"
	@echo "  perf-test-gpu     - Run GPU performance tests with cuDF/RAPIDS"
	@echo "  perf-test-all     - Run both CPU and GPU performance tests"
	@echo ""
	@echo "Usage examples:"
	@echo "  make"
	@echo "  make run"
	@echo "  make perf-test-cpu"
	@echo "  make perf-test-gpu"
	@echo "  make perf-test-all"
	@echo "  ./bin/HW3_MCC_030402_401106039 --mode cpu --rules rules.txt --input set1.txt --threads 4"
	@echo "  ./bin/HW3_MCC_030402_401106039 --mode gpu --rules rules.txt --input set1.txt"
