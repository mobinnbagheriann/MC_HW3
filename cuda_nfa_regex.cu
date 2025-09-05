// Build: nvcc -std=c++17 -O2 cuda_regex_nfa.cu -o cuda_regex_nfa
// Run:   ./cuda_regex_nfa sentences.txt patterns.txt
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <stack>
#include <limits>
#include <algorithm>

#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)


static constexpr int MAX_SENTENCE_LENGTH     = 2048;  // bytes incl. '\0' guard in buffers
static constexpr int MAX_PATTERNS            = 4096;  // safeguard
static constexpr int MAX_SENTENCES           = 131072;// safeguard
static constexpr int MAX_STATES_PER_PATTERN  = 512;   // Thompson NFA state cap (per pattern)
static constexpr int MAX_TOTAL_STATES        = 1 << 20; // total across all patterns (safety)


// Thompson NFA states. We store them contiguous per pattern on device.
enum StateKind : int { ST_CHAR = 0, ST_ANY = 1, ST_SPLIT = 2, ST_MATCH = 3, ST_CLASS_DIGIT = 4 };

struct NFAState {
  int kind;   // StateKind
  int c;      // for ST_CHAR: literal char; unused otherwise
  int out1;   // index of first out state, -1 if none
  int out2;   // index of second out state (for SPLIT), -1 otherwise
};

struct PatternMeta {
  int offset;        // offset into global states array
  int nstates;       // number of states for this pattern
  unsigned char anchored_start; // 1 if ^ present at pattern start
  unsigned char anchored_end;   // 1 if $ present at pattern end
  unsigned char _pad[2];
};

// Token kinds for infix -> postfix conversion (shunting-yard)
enum TokKind { TK_CHAR, TK_ANY, TK_ALT, TK_LPAREN, TK_RPAREN, TK_STAR, TK_PLUS, TK_QMARK, TK_CONCAT, TK_CLASS_DIGIT };

struct Token {
  TokKind k;
  int ch; // for TK_CHAR
};

static inline bool is_regex_metachar(char c) {
  switch (c) {
    case '.': case '|': case '(': case ')': case '*': case '+': case '?': case '^': case '$':
      return true;
    default: return false;
  }
}

static std::vector<Token> tokenize_and_prepare(const std::string& pattern_raw, bool& anchored_start, bool& anchored_end) {
  anchored_start = false;
  anchored_end   = false;

  // Handle ^ and $ only if at edges
  size_t start = 0, end = pattern_raw.size();
  if (start < end && pattern_raw[start] == '^') { anchored_start = true; start++; }
  if (end > start && pattern_raw[end - 1] == '$') { anchored_end = true; end--; }

  std::vector<Token> toks;
  toks.reserve((end - start) * 2);

  // Build tokens from pattern[start:end)
  for (size_t i = start; i < end; ++i) {
    char c = pattern_raw[i];
    if (c == '\\') {
      if (i + 1 >= end) throw std::runtime_error("Dangling escape at end of pattern: " + pattern_raw);
      char e = pattern_raw[++i];
      switch (e) {
        case 'd': toks.push_back({TK_CLASS_DIGIT, 0}); break; // \d
        case 't': toks.push_back({TK_CHAR, '\t'}); break;
        case 'n': toks.push_back({TK_CHAR, '\n'}); break;
        case 'r': toks.push_back({TK_CHAR, '\r'}); break;

        case '\\': case '.': case '|': case '(': case ')': case '*': case '+': case '?': case '^': case '$':
          toks.push_back({TK_CHAR, e}); break;
        default:
          toks.push_back({TK_CHAR, e}); break;
      }
    } else if (c == '.') {
      toks.push_back({TK_ANY, 0});
    } else if (c == '|') {
      toks.push_back({TK_ALT, 0});
    } else if (c == '(') {
      toks.push_back({TK_LPAREN, 0});
    } else if (c == ')') {
      toks.push_back({TK_RPAREN, 0});
    } else if (c == '*') {
      toks.push_back({TK_STAR, 0});
    } else if (c == '+') {
      toks.push_back({TK_PLUS, 0});
    } else if (c == '?') {
      toks.push_back({TK_QMARK, 0});
    } else {
      toks.push_back({TK_CHAR, (unsigned char)c});
    }
  }


  auto should_concat = [](TokKind a, TokKind b)->bool {

    auto is_atom_a = (a == TK_CHAR || a == TK_ANY || a == TK_CLASS_DIGIT || a == TK_RPAREN || a == TK_STAR || a == TK_PLUS || a == TK_QMARK);
    auto is_atom_b = (b == TK_CHAR || b == TK_ANY || b == TK_CLASS_DIGIT || b == TK_LPAREN);
    return is_atom_a && is_atom_b;
  };

  std::vector<Token> with_concat;
  for (size_t i = 0; i < toks.size(); ++i) {
    with_concat.push_back(toks[i]);
    if (i + 1 < toks.size() && should_concat(toks[i].k, toks[i + 1].k)) {
      with_concat.push_back({TK_CONCAT, 0});
    }
  }
  return with_concat;
}

static int precedence(TokKind k) {
  switch (k) {
    case TK_STAR:
    case TK_PLUS:
    case TK_QMARK: return 3;
    case TK_CONCAT: return 2;
    case TK_ALT:    return 1;
    default:        return 0;
  }
}

static std::vector<Token> infix_to_postfix(const std::vector<Token>& in) {
  std::vector<Token> out;
  std::vector<TokKind> ops;
  for (auto t : in) {
    switch (t.k) {
      case TK_CHAR:
      case TK_ANY:
      case TK_CLASS_DIGIT:
        out.push_back(t); break;
      case TK_LPAREN:
        ops.push_back(TK_LPAREN); break;
      case TK_RPAREN:
        while (!ops.empty() && ops.back() != TK_LPAREN) {
          out.push_back({ops.back(), 0});
          ops.pop_back();
        }
        if (ops.empty() || ops.back() != TK_LPAREN) throw std::runtime_error("Mismatched parentheses");
        ops.pop_back(); // pop '('
        break;
      case TK_ALT:
      case TK_CONCAT:
      case TK_STAR:
      case TK_PLUS:
      case TK_QMARK: {
        int p = precedence(t.k);
        while (!ops.empty()) {
          TokKind top = ops.back();
          if (top == TK_LPAREN) break;
          if (precedence(top) >= p) {
            out.push_back({top, 0});
            ops.pop_back();
          } else break;
        }
        ops.push_back(t.k);
      } break;
    }
  }
  while (!ops.empty()) {
    if (ops.back() == TK_LPAREN) throw std::runtime_error("Mismatched parentheses");
    out.push_back({ops.back(), 0});
    ops.pop_back();
  }
  return out;
}

// Thompson NFA fragment
struct Frag {
  int start;                 // start state index
  std::vector<int> outlist;  // list of state indices whose out1/out2 need patching to next
};

// Helper to append out lists
static std::vector<int> append_out(const std::vector<int>& a, const std::vector<int>& b) {
  std::vector<int> r; r.reserve(a.size() + b.size());
  r.insert(r.end(), a.begin(), a.end());
  r.insert(r.end(), b.begin(), b.end());
  return r;
}

static void patch(std::vector<NFAState>& st, const std::vector<int>& outlist, int target) {
  for (int idx : outlist) {
    // idx points to a "dangling" out1 or out2 encoded as negative slot;
    // but we store "dangling" by using out1==-1 and we remember the state index itself
    // here we assume dangling is always out1 when building sequentially.
    if (st[idx].out1 == -1) st[idx].out1 = target;
    else if (st[idx].out2 == -1) st[idx].out2 = target;
    else throw std::runtime_error("patch: no available out to patch");
  }
}

// Allocate a state and return index
static int add_state(std::vector<NFAState>& st, StateKind k, int c, int o1, int o2) {
  if ((int)st.size() >= MAX_STATES_PER_PATTERN) throw std::runtime_error("Pattern exceeds max NFA states");
  st.push_back({(int)k, c, o1, o2});
  return (int)st.size() - 1;
}

// Build NFA from postfix tokens
static void build_nfa_from_postfix(const std::vector<Token>& postfix, std::vector<NFAState>& out_states) {
  std::vector<Frag> stack;
  stack.reserve(postfix.size());

  auto push_char = [&](StateKind kind, int ch) {
    int s = add_state(out_states, kind, ch, -1, -1);
    Frag f { s, { s } };
    stack.push_back(std::move(f));
  };

  for (auto t : postfix) {
    switch (t.k) {
      case TK_CHAR:        push_char(ST_CHAR, t.ch); break;
      case TK_ANY:         push_char(ST_ANY, 0); break;
      case TK_CLASS_DIGIT: push_char(ST_CLASS_DIGIT, 0); break;

      case TK_CONCAT: {
        if (stack.size() < 2) throw std::runtime_error("Bad concat");
        Frag f2 = stack.back(); stack.pop_back();
        Frag f1 = stack.back(); stack.pop_back();
        patch(out_states, f1.outlist, f2.start);
        Frag f { f1.start, f2.outlist };
        stack.push_back(std::move(f));
      } break;

      case TK_ALT: {
        if (stack.size() < 2) throw std::runtime_error("Bad alternation");
        Frag f2 = stack.back(); stack.pop_back();
        Frag f1 = stack.back(); stack.pop_back();
        int s = add_state(out_states, ST_SPLIT, 0, f1.start, f2.start);
        Frag f { s, append_out(f1.outlist, f2.outlist) };
        stack.push_back(std::move(f));
      } break;

      case TK_STAR: {
        if (stack.empty()) throw std::runtime_error("Bad star");
        Frag f = stack.back(); stack.pop_back();
        int s = add_state(out_states, ST_SPLIT, 0, f.start, -1);
        patch(out_states, f.outlist, s);
        Frag fnew { s, { s } };
        stack.push_back(std::move(fnew));
      } break;

      case TK_PLUS: {
        if (stack.empty()) throw std::runtime_error("Bad plus");
        Frag f = stack.back(); stack.pop_back();
        int s = add_state(out_states, ST_SPLIT, 0, f.start, -1);
        patch(out_states, f.outlist, s);
        Frag fnew { f.start, { s } };
        stack.push_back(std::move(fnew));
      } break;

      case TK_QMARK: {
        if (stack.empty()) throw std::runtime_error("Bad question");
        Frag f = stack.back(); stack.pop_back();
        int s = add_state(out_states, ST_SPLIT, 0, f.start, -1);
        Frag fnew { s, append_out(f.outlist, { s }) };
        stack.push_back(std::move(fnew));
      } break;

      default:
        throw std::runtime_error("Unsupported token in postfix");
    }
  }

  if (stack.size() != 1) throw std::runtime_error("Bad postfix stack");
  Frag f = stack.back(); stack.pop_back();

  int m = add_state(out_states, ST_MATCH, 0, -1, -1);
  patch(out_states, f.outlist, m);
}

// Build one pattern: tokenize -> postfix -> NFA
static std::vector<NFAState> compile_pattern(const std::string& raw,
                                             bool& anchored_start,
                                             bool& anchored_end) {
  auto in = tokenize_and_prepare(raw, anchored_start, anchored_end);
  auto post = infix_to_postfix(in);
  std::vector<NFAState> states;
  states.reserve(64);
  build_nfa_from_postfix(post, states);
  return states;
}


static std::vector<std::string> read_lines_or_die(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open: " + path);
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    lines.push_back(line);
  }
  return lines;
}

static void copy_strings_fixed_width(const std::vector<std::string>& v, char* dst, int maxLen) {
  for (size_t i = 0; i < v.size(); ++i) {
    size_t n = std::min(v[i].size(), (size_t)maxLen - 1);
    std::memcpy(dst + i * maxLen, v[i].data(), n);
    dst[i * maxLen + n] = '\0';
    if (n + 1 < (size_t)maxLen)
      std::memset(dst + i * maxLen + n + 1, 0, maxLen - (int)n - 1);
  }
}


__device__ inline bool is_digit(char c) {
  return c >= '0' && c <= '9';
}

__device__ bool state_matches_char(const NFAState& st, char c) {
  if (st.kind == ST_CHAR) return c == (char)st.c;
  if (st.kind == ST_ANY)  return c != '\0';
  if (st.kind == ST_CLASS_DIGIT) return is_digit(c);
  return false;
}

// Small set for active states (indices), with visited bitmap to avoid repeats
template<int MAXS>
struct StateSet {
  int items[MAXS];
  int count;
  unsigned char seen[MAXS];
  __device__ void clear() {
    count = 0;
    for (int i = 0; i < MAXS; ++i) seen[i] = 0;
  }
  __device__ void add(int s) {
    if (!seen[s]) { seen[s] = 1; items[count++] = s; }
  }
};


template<int MAXS>
__device__ void add_epsilon_closure(const NFAState* states, int nstates, StateSet<MAXS>& set, int s) {

  int stack_local[MAXS];
  int sp = 0;
  if (s >= 0 && s < nstates) stack_local[sp++] = s;
  while (sp) {
    int x = stack_local[--sp];
    const NFAState& st = states[x];
    if (st.kind == ST_SPLIT) {
      if (st.out1 >= 0 && st.out1 < nstates && !set.seen[st.out1]) { set.seen[st.out1] = 1; stack_local[sp++] = st.out1; }
      if (st.out2 >= 0 && st.out2 < nstates && !set.seen[st.out2]) { set.seen[st.out2] = 1; stack_local[sp++] = st.out2; }
      if (!set.seen[x]) { set.seen[x] = 1; set.items[set.count++] = x; }
    } else {
      if (!set.seen[x]) { set.seen[x] = 1; set.items[set.count++] = x; }
    }
  }
}

template<int MAXS>
__device__ bool nfa_search(const NFAState* states, int nstates,
                           const char* text, int tlen,
                           bool anchored_start, bool anchored_end)
{
  int start_pos_begin = anchored_start ? 0 : 0;
  int start_pos_end   = anchored_start ? 0 : tlen;

  for (int start_pos = start_pos_begin; start_pos <= start_pos_end; ++start_pos) {
    StateSet<MAXS> curr, next;
    curr.clear(); next.clear();
    add_epsilon_closure<MAXS>(states, nstates, curr, 0);


    for (int i = start_pos; i < tlen; ++i) {
      char c = text[i];
      next.clear();
      for (int k = 0; k < curr.count; ++k) {
        int si = curr.items[k];
        const NFAState& st = states[si];
        if (st.kind == ST_CHAR || st.kind == ST_ANY || st.kind == ST_CLASS_DIGIT) {
          if (state_matches_char(st, c)) {
            add_epsilon_closure<MAXS>(states, nstates, next, st.out1);
          }
        } else if (st.kind == ST_MATCH) {
          if (!anchored_end) return true;
        }
      }
      curr = next;
      if (curr.count == 0) break; // dead
    }

    for (int k = 0; k < curr.count; ++k) {
      const NFAState& st = states[curr.items[k]];
      if (st.kind == ST_MATCH) {
        return true;
      }
    }

    if (anchored_start) break;
  }
  return false;
}

// Kernel: 2D grid (x: sentence, y: pattern)
__global__ void regex_kernel_2d(
    const char* __restrict__ sentences,
    const int*  __restrict__ sent_lengths,
    int numSentences,
    int maxSentenceLen,
    const NFAState* __restrict__ all_states,
    const PatternMeta* __restrict__ metas,
    int numPatterns,
    int* __restrict__ out_counts)
{
  int sIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int pIdx = blockIdx.y * blockDim.y + threadIdx.y;
  if (sIdx >= numSentences || pIdx >= numPatterns) return;

  const char* s = sentences + sIdx * maxSentenceLen;
  int slen = sent_lengths[sIdx];

  PatternMeta pm = metas[pIdx];
  const NFAState* st = all_states + pm.offset;
  int nst = pm.nstates;

  bool ok = nfa_search<MAX_STATES_PER_PATTERN>(st, nst, s, slen,
                                               pm.anchored_start != 0,
                                               pm.anchored_end   != 0);
  if (ok) atomicAdd(&out_counts[sIdx], 1);
}


int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "Usage: " << argv[0] << " <sentences.txt> <patterns.txt>\n";
      return EXIT_FAILURE;
    }

    auto sentences = read_lines_or_die(argv[1]);
    auto patterns_raw = read_lines_or_die(argv[2]);
    if (sentences.empty()) throw std::runtime_error("Sentences file is empty");
    if (patterns_raw.empty()) throw std::runtime_error("Patterns file is empty");
    if ((int)sentences.size() > MAX_SENTENCES) throw std::runtime_error("Too many sentences");
    if ((int)patterns_raw.size() > MAX_PATTERNS) throw std::runtime_error("Too many patterns");

    std::vector<NFAState> all_states_host;
    all_states_host.reserve(std::min(1<<18, MAX_TOTAL_STATES));
    std::vector<PatternMeta> metas_host;
    metas_host.reserve(patterns_raw.size());

    int total_states = 0;
    for (size_t i = 0; i < patterns_raw.size(); ++i) {
      bool astart=false, aend=false;
      auto st = compile_pattern(patterns_raw[i], astart, aend);
      if (st.empty()) throw std::runtime_error("Failed to build NFA (empty)");
      if (total_states + (int)st.size() > MAX_TOTAL_STATES) {
        throw std::runtime_error("Total NFA states exceed MAX_TOTAL_STATES");
      }
      PatternMeta pm{};
      pm.offset = total_states;
      pm.nstates = (int)st.size();
      pm.anchored_start = astart ? 1 : 0;
      pm.anchored_end   = aend   ? 1 : 0;
      metas_host.push_back(pm);

      all_states_host.insert(all_states_host.end(), st.begin(), st.end());
      total_states += (int)st.size();
    }

    const int numSentences = (int)sentences.size();
    const int numPatterns  = (int)patterns_raw.size();
    size_t sentencesBytes = (size_t)numSentences * MAX_SENTENCE_LENGTH;
    std::vector<char> h_sentences(sentencesBytes, 0);
    std::vector<int>  h_slen(numSentences, 0);

    for (int i = 0; i < numSentences; ++i) {
      auto& s = sentences[i];
      size_t n = std::min(s.size(), (size_t)MAX_SENTENCE_LENGTH - 1);
      std::memcpy(&h_sentences[i * MAX_SENTENCE_LENGTH], s.data(), n);
      h_sentences[i * MAX_SENTENCE_LENGTH + n] = '\0';
      h_slen[i] = (int)n;
    }

    char* d_sentences = nullptr;
    int*  d_slen = nullptr;
    NFAState* d_states = nullptr;
    PatternMeta* d_meta = nullptr;
    int* d_counts = nullptr;

    CUDA_CHECK(cudaMalloc(&d_sentences, sentencesBytes));
    CUDA_CHECK(cudaMalloc(&d_slen, numSentences * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states, all_states_host.size() * sizeof(NFAState)));
    CUDA_CHECK(cudaMalloc(&d_meta, metas_host.size() * sizeof(PatternMeta)));
    CUDA_CHECK(cudaMalloc(&d_counts, numSentences * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_counts, 0, numSentences * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_sentences, h_sentences.data(), sentencesBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slen, h_slen.data(), numSentences * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_states, all_states_host.data(), all_states_host.size() * sizeof(NFAState), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_meta, metas_host.data(), metas_host.size() * sizeof(PatternMeta), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid( (numSentences + block.x - 1) / block.x,
               (numPatterns  + block.y - 1) / block.y );

    regex_kernel_2d<<<grid, block>>>(
      d_sentences, d_slen, numSentences, MAX_SENTENCE_LENGTH,
      d_states, d_meta, numPatterns, d_counts
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_counts(numSentences, 0);
    CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, numSentences * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < numSentences; ++i) {
      std::cout << h_counts[i] << "\n";
    }

    cudaFree(d_sentences);
    cudaFree(d_slen);
    cudaFree(d_states);
    cudaFree(d_meta);
    cudaFree(d_counts);

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return EXIT_FAILURE;
  }
}

