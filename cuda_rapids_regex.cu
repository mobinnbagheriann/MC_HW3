// Build: mkdir build && cd build && cmake .. && cmake --build . -j
// Run:   ./cuda_rapids_regex sentences.txt patterns.txt
#include <cuda_runtime.h>
#include <rmm/device_uvector.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/types.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { \
  cudaError_t _e = (x); \
  if (_e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(_e) \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

static std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open: " + path);
  std::vector<std::string> lines;
  std::string s;
  while (std::getline(in, s)) lines.push_back(s);
  return lines;
}

// Build device strings column from host vector<string>
// using offsets + concatenated chars, then cudf::make_strings_column
static std::unique_ptr<cudf::column>
make_device_strings(const std::vector<std::string>& h, rmm::cuda_stream_view stream) {
  using size_type = cudf::size_type;
  const size_type n = static_cast<size_type>(h.size());

  std::vector<int32_t> h_offsets(n + 1, 0);
  size_t total_chars = 0;
  for (size_t i = 0; i < h.size(); ++i) {
    total_chars += h[i].size();
    h_offsets[i + 1] = static_cast<int32_t>(total_chars);
  }
  std::vector<char> h_chars;
  h_chars.reserve(total_chars);
  for (auto& s : h) h_chars.insert(h_chars.end(), s.begin(), s.end());

  rmm::device_uvector<int32_t> d_offsets(n + 1, stream);
  rmm::device_uvector<char>    d_chars(total_chars, stream);

  CUDA_CHECK(cudaMemcpyAsync(d_offsets.data(), h_offsets.data(),
                             (n + 1) * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream.value()));
  if (total_chars) {
    CUDA_CHECK(cudaMemcpyAsync(d_chars.data(), h_chars.data(), total_chars,
                               cudaMemcpyHostToDevice, stream.value()));
  }

  auto null_mask = rmm::device_buffer{0, stream};
  auto null_count = 0;

  return cudf::make_strings_column(
      n, std::move(d_offsets), d_chars.release(), null_count, std::move(null_mask));
}

__global__ void add_true_to_counts(const uint8_t* __restrict__ vals,
                                   int n,
                                   int* __restrict__ counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) counts[i] += (vals[i] != 0);
}

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "Usage: " << argv[0] << " <sentences.txt> <patterns.txt>\n";
      return EXIT_FAILURE;
    }

    auto sentences = read_lines(argv[1]);
    auto patterns  = read_lines(argv[2]);
    if (sentences.empty()) throw std::runtime_error("Sentences file is empty.");
    if (patterns.empty())  throw std::runtime_error("Patterns file is empty.");

    auto stream = rmm::cuda_stream_default;
    auto sentences_col = make_device_strings(sentences, stream);
    cudf::strings_column_view sview{sentences_col->view()};
    const int nrows = static_cast<int>(sview.size());

    rmm::device_uvector<int> d_counts(nrows, stream);
    CUDA_CHECK(cudaMemsetAsync(d_counts.data(), 0, nrows * sizeof(int), stream.value()));

    for (const auto& pat : patterns) {
      auto prog = cudf::strings::regex_program::create(pat);
      auto bool_col = cudf::strings::contains_re(sview, prog);

      auto bv = bool_col->view();
      const uint8_t* d_vals = bv.data<uint8_t>();

      int threads = 256;
      int blocks = (nrows + threads - 1) / threads;
      add_true_to_counts<<<blocks, threads, 0, stream.value()>>>(d_vals, nrows, d_counts.data());
      CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaStreamSynchronize(stream.value()));

    std::vector<int> h_counts(nrows);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts.data(),
                          nrows * sizeof(int), cudaMemcpyDeviceToHost));
    for (int c : h_counts) std::cout << c << "\n";
    return EXIT_SUCCESS;

  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}

