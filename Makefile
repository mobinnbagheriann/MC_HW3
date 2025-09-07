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
#
# This version keeps your original targets and names intact, but:
#  - auto-detects RAPIDS/cuDF from the active conda env (CONDA_PREFIX)
#  - auto-detects CUDA from the nvcc on PATH
#  - links Hyperscan via pkg-config if available, else falls back to /usr/local[/hyperscan]
#  - sets GPU arch to sm_75 for NVIDIA T4
#
# ---------------------------------------------------------------------------
# Toolchain
NVCC       := nvcc
NVCCFLAGS  := -O3 -std=c++17 -arch=sm_75 -Xcompiler -Wall,-Wextra,-fPIC

# ---------------------------------------------------------------------------
# Include paths for RAPIDS/cuDF (auto) and CUDA (auto)
RAPIDS_ROOT ?= $(CONDA_PREFIX)
ifeq ($(strip $(RAPIDS_ROOT)),)
  $(error No conda environment active. Run: 'conda activate rapids-25.08')
endif

# Discover CUDA root from nvcc location (works for conda-installed cudatoolkit too)
CUDA_ROOT ?= $(shell dirname $$(dirname $$(which nvcc)))

# ---------------------------------------------------------------------------
# Hyperscan detection (prefer pkg-config; fallback to common prefixes)
HS_PKG_CFLAGS := $(shell pkg-config --cflags libhs 2>/dev/null)
HS_PKG_LIBS   := $(shell pkg-config --libs libhs 2>/dev/null)
HS_PREFIX ?= $(shell if [ -d $(CONDA_PREFIX)/include/hs ]; then echo $(CONDA_PREFIX); elif [ -d /usr/local/hyperscan ]; then echo /usr/local/hyperscan; else echo /usr/local; fi)
HS_INC := $(if $(HS_PKG_CFLAGS),$(HS_PKG_CFLAGS),-I$(HS_PREFIX)/include)
HS_LIB := $(if $(HS_PKG_LIBS),$(HS_PKG_LIBS),-L$(HS_PREFIX)/lib -lhs)

# ---------------------------------------------------------------------------
# Include & link flags
INCLUDES := -I$(RAPIDS_ROOT)/include \
           -I$(CUDA_ROOT)/include \
           -I$(RAPIDS_ROOT)/include/libcudf/libcudacxx \
           $(HS_INC)

LDFLAGS := -L$(RAPIDS_ROOT)/lib \
          -L$(CUDA_ROOT)/lib64 \
          -L/usr/lib/x86_64-linux-gnu \
          -Wl,-rpath,$(RAPIDS_ROOT)/lib \
          -Wl,-rpath,$(CUDA_ROOT)/lib64 \
          $(HS_LIB) \
          -lcudf -lrmm -lcudart -lpthread

# ---------------------------------------------------------------------------
# Project layout (kept as in your file)
SRC_DIR      := src
BIN_DIR      := bin
RESULTS_DIR  := results

# Using the format from the assignment PDF (kept unchanged)
TARGET := $(BIN_DIR)/HW3_MCC_030402_401106039
SRC    := $(SRC_DIR)/main.cu

# ---------------------------------------------------------------------------
# Default example parameters (handy for "make run")
DEFAULT_RULES   := rules.txt
DEFAULT_INPUT   := set1.txt
DEFAULT_THREADS := 4
DEFAULT_MODE    := cpu

# ---------------------------------------------------------------------------
# Build rules (kept)
.PHONY: all debug clean run perf-test-cpu perf-test-gpu perf-test-all help

all: $(TARGET)

debug: NVCCFLAGS := -g -G -O0 -std=c++17 -arch=sm_75 -Xcompiler -Wall,-Wextra,-fPIC
debug: INCLUDES  := -I$(RAPIDS_ROOT)/include -I$(CUDA_ROOT)/include -I$(RAPIDS_ROOT)/include/libcudf/libcudacxx $(HS_INC)
debug: LDFLAGS   := -L$(RAPIDS_ROOT)/lib -L$(CUDA_ROOT)/lib64 -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,$(RAPIDS_ROOT)/lib -Wl,-rpath,$(CUDA_ROOT)/lib64 $(HS_LIB) -lcudf -lrmm -lcudart -lpthread
debug: clean $(TARGET)

$(TARGET): $(SRC) |
