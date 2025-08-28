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
CC         := gcc
# NVCC       := nvcc  # Commented out - will be enabled when CUDA is available
# Add -lhs for the Hyperscan library, -lcudart for CUDA (commented out for now)
CFLAGS     := -Wall -Wextra -std=c11 -O3 -pthread
LDFLAGS    := -pthread -lhs
# LDFLAGS    := -pthread -lhs -lcudart  # Uncomment when CUDA is available
# NVCCFLAGS  := -O3  # Commented out - will be enabled when CUDA is available

# ---------------------------------------------------------------------------
# Project layout
# NOTE: Your source code (main.c) should be placed inside the 'src' directory.
SRC_DIR      := src
BIN_DIR      := bin
RESULTS_DIR  := results

# Using the format from the assignment PDF
TARGET := $(BIN_DIR)/HW3_MCC_030402_401106039
SRC    := $(SRC_DIR)/main.c

# ---------------------------------------------------------------------------
# Default example parameters (handy for "make run")
# NOTE: Create these placeholder files for a quick test.
DEFAULT_RULES := rules.txt
DEFAULT_INPUT := set1.txt
DEFAULT_THREADS := 4
DEFAULT_MODE := cpu

# ---------------------------------------------------------------------------
# Build rules
.PHONY: all debug clean run perf-test-cpu perf-test-gpu help

all: $(TARGET)

debug: CFLAGS := -Wall -Wextra -std=c11 -g -O0 -pthread
debug: LDFLAGS := -pthread -lhs
debug: clean all

$(TARGET): $(SRC) | $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR):
	mkdir -p $@

# The 'run' and 'perf-test' targets will create this directory
$(RESULTS_DIR):
	mkdir -p $@

clean:
	@echo "Cleaning up build artifacts and results..."
	rm -rf $(BIN_DIR)/*
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
	@echo "GPU performance tests complete."

# ---------------------------------------------------------------------------
# Help
help:
	@echo "Available targets:"
	@echo "  all              - Build the program with optimizations (default)"
	@echo "  debug            - Build with debug symbols and no optimization"
	@echo "  clean            - Remove build artifacts and result files"
	@echo "  run              - Run a quick demo with default settings"
	@echo "  perf-test-cpu    - Run a performance sweep for the CPU version"
	@echo "  perf-test-gpu    - Placeholder for the GPU performance test"
	@echo "  help             - Show this help message"

