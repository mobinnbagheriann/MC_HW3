#define _GNU_SOURCE  // For getline and strdup
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <hs/hs.h>
#include <sys/stat.h>
#include <unistd.h>

// CUDA includes for GPU mode
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <regex>
#include <vector>
#include <algorithm>


// --- Data Structures ---

// Execution Mode
typedef enum {
    MODE_CPU,
    MODE_GPU
} execution_mode_t;

// Configuration Structure
typedef struct {
    execution_mode_t mode;
    char* rules_file;
    char* input_file;
    int num_threads;  // Only used for CPU mode
} config_t;

/**
 * @struct MatchContext
 * @brief Context structure passed to the Hyperscan match event handler.
 */
typedef struct {
    int* matches;           // Array to store IDs of matched rules.
    int match_count;        // Number of matches found for the current line.
    int match_capacity;     // Allocated capacity of the matches array.
} MatchContext;

/**
 * @struct ThreadData
 * @brief Data structure to pass information to each worker thread.
 */
typedef struct {
    int thread_id;                 // Unique identifier for the thread.
    char** lines;                  // Pointer to the array of all input lines.
    unsigned int* line_lengths;    // Pointer to the array of all line lengths.
    long start_line;               // Starting line index for this thread.
    long end_line;                 // Ending line index for this thread.
    hs_database_t* database;       // Pointer to the compiled Hyperscan database.
    hs_scratch_t* scratch;         // Per-thread scratch space for Hyperscan.
    char*** thread_results;        // 2D array: [line_index][match_list] for this thread's lines
    long total_matches;            // Total number of matches found by this thread.
} ThreadData;


// --- Forward Declarations ---
int run_cpu_mode(const config_t* config);
int run_gpu_mode(const config_t* config);

// --- GPU Helper Functions ---

/**
 * @brief Check CUDA errors and exit on failure
 */
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Simple string matching function that will run on GPU
 * This is a simplified regex matcher using basic string matching
 * In a real implementation, you would use a proper GPU regex library
 */
__global__ void gpu_regex_match_kernel(char** d_lines, unsigned int* d_line_lengths, 
                                        char** d_patterns, unsigned int* d_pattern_lengths,
                                        int pattern_count, int* d_results, int* d_match_counts, 
                                        int line_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= line_count) return;
    
    int matches = 0;
    char* line = d_lines[tid];
    unsigned int line_len = d_line_lengths[tid];
    
    // Simple pattern matching (this is a simplified version)
    // In real implementation, you would use proper regex library
    for (int p = 0; p < pattern_count; p++) {
        char* pattern = d_patterns[p];
        unsigned int pattern_len = d_pattern_lengths[p];
        
        // Simple substring search
        for (int i = 0; i <= (int)line_len - (int)pattern_len; i++) {
            bool match = true;
            for (int j = 0; j < (int)pattern_len; j++) {
                if (line[i + j] != pattern[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                // Store pattern ID in results array
                d_results[tid * pattern_count + matches] = p;
                matches++;
                break; // Move to next pattern
            }
        }
    }
    
    d_match_counts[tid] = matches;
}


// --- Utility Functions ---

/**
 * @brief Prints an error message and exits the program.
 */
void fail(const char* msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(EXIT_FAILURE);
}


/**
 * @brief Print usage information.
 */
void print_usage(const char* program_name) {
    printf("Usage: %s --mode <cpu|gpu> --rules <rules_file> --input <input_file> [--threads <num_threads>]\n", program_name);
    printf("\nRequired arguments:\n");
    printf("  --mode      <cpu|gpu>       Processing mode (CPU or GPU)\n");
    printf("  --rules     <rules_file>    Path to the rules file\n");
    printf("  --input     <input_file>    Path to the input file\n");
    printf("\nOptional arguments:\n");
    printf("  --threads   <num_threads>   Number of threads (required for CPU mode)\n");
    printf("\nOutput files are automatically generated in the results/ directory:\n");
    printf("  Results_HW3_MCC_030402_401106039_{CPU/GPU}_{DataSet}_{NumThreads/Library}.txt\n");
    printf("  Results_HW3_MCC_030402_401106039_{CPU/GPU}_{DataSet}_{Hyperscan/GPULibrary}.csv\n");
    printf("\nExample:\n");
    printf("  %s --mode cpu --rules rules.txt --input set1.txt --threads 4\n", program_name);
    printf("  %s --mode gpu --rules rules.txt --input set1.txt\n", program_name);
    exit(EXIT_FAILURE);
}

/**
 * @brief Generate automatic output filename based on configuration.
 */
char* generate_output_filename(const config_t* config) {
    // Extract dataset name from input file (e.g., "set1.txt" -> "set1")
    const char* input_basename = strrchr(config->input_file, '/');
    if (input_basename) {
        input_basename++; // Skip the '/'
    } else {
        input_basename = config->input_file;
    }
    
    // Remove file extension
    char dataset[256];
    strncpy(dataset, input_basename, sizeof(dataset) - 1);
    dataset[sizeof(dataset) - 1] = '\0';
    char* dot = strrchr(dataset, '.');
    if (dot) {
        *dot = '\0';
    }
    
    // Allocate memory for the filename
    char* filename = (char*)malloc(512);
    if (!filename) {
        fprintf(stderr, "Error: Memory allocation failed for output filename\n");
        exit(EXIT_FAILURE);
    }
    
    if (config->mode == MODE_CPU) {
        snprintf(filename, 512, "results/Results_HW3_MCC_030402_401106039_CPU_%s_%d.txt", 
                 dataset, config->num_threads);
    } else {
        snprintf(filename, 512, "results/Results_HW3_MCC_030402_401106039_GPU_%s_CUDA.txt", 
                 dataset);
    }
    
    return filename;
}

/**
 * @brief Generate performance CSV filename based on configuration.
 */
char* generate_performance_filename(const config_t* config, const char* input_filename) {
    // Extract dataset name from input filename
    const char* dataset_name = strrchr(input_filename, '/');
    if (dataset_name) {
        dataset_name++; // Skip the '/'
    } else {
        dataset_name = input_filename;
    }
    
    // Remove extension from dataset name
    char* dataset_clean = strdup(dataset_name);
    char* dot = strrchr(dataset_clean, '.');
    if (dot) *dot = '\0';
    
    char* filename = (char*)malloc(512);
    if (config->mode == MODE_CPU) {
        snprintf(filename, 512, "results/Results_HW3_MCC_030402_401106039_CPU_%s_Hyperscan.csv", 
                 dataset_clean);
    } else {
        snprintf(filename, 512, "results/Results_HW3_MCC_030402_401106039_GPU_%s_CUDA.csv", 
                 dataset_clean);
    }
    
    free(dataset_clean);
    return filename;
}

/**
 * @brief Parse command line arguments.
 */
config_t parse_arguments(int argc, char* argv[]) {
    config_t config = {0};
    
    if (argc < 7) {  // Minimum required arguments
        print_usage(argv[0]);
    }
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            if (strcmp(argv[i + 1], "cpu") == 0) {
                config.mode = MODE_CPU;
            } else if (strcmp(argv[i + 1], "gpu") == 0) {
                config.mode = MODE_GPU;
            } else {
                fprintf(stderr, "ERROR: Invalid mode '%s'. Use 'cpu' or 'gpu'.\n", argv[i + 1]);
                print_usage(argv[0]);
            }
            i++; // Skip next argument
        } else if (strcmp(argv[i], "--rules") == 0 && i + 1 < argc) {
            config.rules_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            config.input_file = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            config.num_threads = atoi(argv[i + 1]);
            if (config.num_threads <= 0) {
                fprintf(stderr, "ERROR: Number of threads must be a positive integer.\n");
                print_usage(argv[0]);
            }
            i++;
        }
    }
    
    // Validate required arguments
    if (!config.rules_file || !config.input_file) {
        fprintf(stderr, "ERROR: Missing required arguments.\n");
        print_usage(argv[0]);
    }
    
    if (config.mode == MODE_CPU && config.num_threads == 0) {
        fprintf(stderr, "ERROR: --threads argument is required for CPU mode.\n");
        print_usage(argv[0]);
    }
    
    return config;
}

/**
 * @brief Reads all lines from a file into a dynamically allocated array.
 */
char** read_lines_from_file(const char* filename, long* line_count, unsigned int** line_lengths, long* total_bytes) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("fopen failed");
        fail("Could not open file.");
    }

    // Get file size for total_bytes metric
    struct stat st;
    if (stat(filename, &st) == 0) {
        *total_bytes = st.st_size;
    } else {
        *total_bytes = 0; // Fallback
    }

    long capacity = 1024;
    char** lines = (char**)malloc(capacity * sizeof(char*));
    if (!lines) fail("Failed to allocate memory for lines.");

    *line_count = 0;
    char* line_buffer = NULL;
    size_t buffer_size = 0;

    while (getline(&line_buffer, &buffer_size, file) != -1) {
        if (*line_count >= capacity) {
            capacity *= 2;
            lines = (char**)realloc(lines, capacity * sizeof(char*));
            if (!lines) fail("Failed to reallocate memory for lines.");
        }
        // Strip newline characters
        line_buffer[strcspn(line_buffer, "\r\n")] = 0;
        lines[*line_count] = strdup(line_buffer);
        if (!lines[*line_count]) fail("Failed to duplicate line.");
        (*line_count)++;
    }

    free(line_buffer);
    fclose(file);

    // Create the line lengths array
    *line_lengths = (unsigned int*)malloc(*line_count * sizeof(unsigned int));
    if (!*line_lengths) fail("Failed to allocate memory for line lengths.");
    for (long i = 0; i < *line_count; i++) {
        (*line_lengths)[i] = strlen(lines[i]);
    }

    return lines;
}


// --- Hyperscan Match Callback ---

/**
 * @brief Hyperscan match event handler.
 */
static int onMatch(unsigned int id, unsigned long long from, unsigned long long to,
                   unsigned int flags, void* ctx) {
    (void)from;   // Suppress unused parameter warning
    (void)to;     // Suppress unused parameter warning
    (void)flags;  // Suppress unused parameter warning
    
    MatchContext* context = (MatchContext*)ctx;

    // Resize matches array if needed
    if (context->match_count >= context->match_capacity) {
        context->match_capacity *= 2;
        context->matches = (int*)realloc(context->matches, context->match_capacity * sizeof(int));
        if (!context->matches) {
            fail("Failed to reallocate memory for matches in callback.");
        }
    }

    context->matches[context->match_count++] = id;
    return 0; // Continue scanning
}


// --- Worker Thread ---

/**
 * @brief The main function for each worker thread.
 */
void* worker_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    data->total_matches = 0;

    // Allocate scratch space for this thread
    hs_error_t scratch_err = hs_alloc_scratch(data->database, &data->scratch);
    if (scratch_err != HS_SUCCESS) {
        fprintf(stderr, "Thread %d: Failed to allocate scratch space. Error: %d\n", data->thread_id, scratch_err);
        return NULL;
    }

    // Allocate 2D result array for this thread's lines
    long thread_line_count = data->end_line - data->start_line;
    data->thread_results = (char***)malloc(thread_line_count * sizeof(char**));
    if (!data->thread_results) {
        fprintf(stderr, "Thread %d: Failed to allocate thread results array.\n", data->thread_id);
        return NULL;
    }

    for (long i = data->start_line; i < data->end_line; i++) {
        long local_index = i - data->start_line; // Local index within this thread's range
        
        // Initialize context for this line's scan
        MatchContext context;
        context.match_capacity = 16; // Initial capacity
        context.matches = (int*)malloc(context.match_capacity * sizeof(int));
        if (!context.matches) {
             data->thread_results[local_index] = (char**)malloc(sizeof(char*));
             data->thread_results[local_index][0] = strdup(""); // Store empty result on failure
             continue;
        }
        context.match_count = 0;

        // Perform the scan
        hs_error_t err = hs_scan(data->database, data->lines[i], data->line_lengths[i], 0,
                                 data->scratch, onMatch, &context);

        if (err != HS_SUCCESS) {
            free(context.matches);
            data->thread_results[local_index] = (char**)malloc(sizeof(char*));
            data->thread_results[local_index][0] = strdup(""); // Store empty result on error
            continue;
        }

        data->total_matches += context.match_count;

        // Format the result string with ZERO-INDEXED pattern numbers (e.g., "0,3,9")
        if (context.match_count > 0) {
            // A rough estimation for buffer size: 10 chars per match ID + commas
            size_t buffer_size = context.match_count * 10;
            char* result_buffer = (char*)malloc(buffer_size);
            if (!result_buffer) {
                data->thread_results[local_index] = (char**)malloc(sizeof(char*));
                data->thread_results[local_index][0] = strdup("");
            } else {
                int offset = 0;
                for (int j = 0; j < context.match_count; j++) {
                    // Use ZERO-INDEXED pattern numbers (Hyperscan IDs start from 0)
                    offset += snprintf(result_buffer + offset, buffer_size - offset,
                                       "%d%s", context.matches[j], (j == context.match_count - 1) ? "" : ",");
                }
                data->thread_results[local_index] = (char**)malloc(sizeof(char*));
                data->thread_results[local_index][0] = result_buffer;
            }
        } else {
            // If no matches, store an empty string
            data->thread_results[local_index] = (char**)malloc(sizeof(char*));
            data->thread_results[local_index][0] = strdup("");
        }

        free(context.matches);
    }

    // Free scratch space allocated by this thread
    if (data->scratch) {
        hs_free_scratch(data->scratch);
    }

    return NULL;
}


// --- CPU Mode Implementation ---

int run_cpu_mode(const config_t* config) {
    // --- 1. Read and Compile Rules ---
    printf("Reading and compiling regex rules from '%s'...\n", config->rules_file);
    long pattern_count = 0;
    long ignored_total_bytes;
    unsigned int* ignored_lengths;
    char** patterns = read_lines_from_file(config->rules_file, &pattern_count, &ignored_lengths, &ignored_total_bytes);
    free(ignored_lengths);

    unsigned int* ids = (unsigned int*)malloc(pattern_count * sizeof(unsigned int));
    unsigned int* flags = (unsigned int*)malloc(pattern_count * sizeof(unsigned int));
    if (!ids || !flags) fail("Failed to allocate memory for rule IDs/flags.");

    for (long i = 0; i < pattern_count; i++) {
        ids[i] = i; // Hyperscan uses 0-indexed IDs
        flags[i] = 0; // No flags
    }

    hs_database_t* database;
    hs_compile_error_t* compile_err;
    hs_platform_info_t platform;
    
    // Populate platform information for optimal compilation
    hs_error_t platform_err = hs_populate_platform(&platform);
    if (platform_err != HS_SUCCESS) {
        printf("Warning: Could not populate platform info, using default settings.\n");
    }
    
    hs_error_t err = hs_compile_multi((const char* const*)patterns, flags, ids, pattern_count,
                                      HS_MODE_BLOCK, (platform_err == HS_SUCCESS) ? &platform : NULL, 
                                      &database, &compile_err);

    if (err != HS_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to compile pattern: %s\n", compile_err->message);
        hs_free_compile_error(compile_err);
        fail("Hyperscan compilation failed.");
    }
    
    if (!database) {
        fail("Database compilation succeeded but database is NULL.");
    }
    
    printf("Compilation successful. %ld rules loaded.\n", pattern_count);

    // --- 2. Read Input Data ---
    printf("Reading input data from '%s'...\n", config->input_file);
    long line_count = 0;
    long total_bytes = 0;
    unsigned int* line_lengths;
    char** lines = read_lines_from_file(config->input_file, &line_count, &line_lengths, &total_bytes);
    printf("Read %ld lines, total size: %.2f MB.\n", line_count, (double)total_bytes / (1024 * 1024));

    // --- 3. Setup and Run Threads ---
    printf("Processing with %d worker thread(s)...\n", config->num_threads);
    pthread_t* threads = (pthread_t*)malloc(config->num_threads * sizeof(pthread_t));
    ThreadData* thread_data = (ThreadData*)malloc(config->num_threads * sizeof(ThreadData));
    if (!threads || !thread_data) {
        fail("Failed to allocate memory for thread management.");
    }

    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    long lines_per_thread = line_count / config->num_threads;
    long remaining_lines = line_count % config->num_threads;
    long current_line = 0;

    for (int i = 0; i < config->num_threads; i++) {
        thread_data[i].thread_id = i;
        thread_data[i].lines = lines;
        thread_data[i].line_lengths = line_lengths;
        thread_data[i].database = database;
        thread_data[i].thread_results = NULL; // Will be allocated by each thread
        thread_data[i].total_matches = 0;
        thread_data[i].scratch = NULL; // Let each thread allocate its own scratch

        // Distribute lines
        thread_data[i].start_line = current_line;
        long chunk_size = lines_per_thread + (i < remaining_lines ? 1 : 0);
        thread_data[i].end_line = current_line + chunk_size;
        current_line += chunk_size;

        pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]);
    }

    // --- 4. Join Threads and Collect Results ---
    long total_matches = 0;
    for (int i = 0; i < config->num_threads; i++) {
        pthread_join(threads[i], NULL);
        total_matches += thread_data[i].total_matches;
    }

    // --- 5. Merge Thread Results into Final Output Array ---
    char** all_results = (char**)malloc(line_count * sizeof(char*));
    if (!all_results) {
        fail("Failed to allocate memory for final results.");
    }

    // Copy results from each thread's 2D array to the final output array
    for (int i = 0; i < config->num_threads; i++) {
        long thread_line_count = thread_data[i].end_line - thread_data[i].start_line;
        for (long j = 0; j < thread_line_count; j++) {
            long global_index = thread_data[i].start_line + j;
            all_results[global_index] = strdup(thread_data[i].thread_results[j][0]);
            
            // Free the thread's result memory
            free(thread_data[i].thread_results[j][0]);
            free(thread_data[i].thread_results[j]);
        }
        free(thread_data[i].thread_results);
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    printf("Processing completed.\n");

    // --- 6. Calculate Performance Metrics ---
    double elapsed_seconds = (end_time.tv_sec - start_time.tv_sec) +
                             (end_time.tv_nsec - start_time.tv_nsec) / 1e9;

    double throughput_input_per_sec = line_count / elapsed_seconds;
    double throughput_mbytes_per_sec = (total_bytes / (1024.0 * 1024.0)) / elapsed_seconds;
    double throughput_match_per_sec = total_matches / elapsed_seconds;
    double latency_ms = (elapsed_seconds * 1000.0) / line_count;

    printf("Performance Metrics:\n");
    printf("  Total Time: %.4f seconds\n", elapsed_seconds);
    printf("  Total Matches: %ld\n", total_matches);
    printf("  Throughput (Input/sec): %.2f\n", throughput_input_per_sec);
    printf("  Throughput (MBytes/sec): %.2f\n", throughput_mbytes_per_sec);
    printf("  Throughput (Match/sec): %.2f\n", throughput_match_per_sec);
    printf("  Latency (ms/input): %.4f\n", latency_ms);

    // --- 7. Write Output Files ---
    char* output_filename = generate_output_filename(config);
    printf("Writing results to '%s'...\n", output_filename);

    // Write match results
    FILE* out_file = fopen(output_filename, "w");
    if (!out_file) fail("Could not open output file for writing.");
    for (long i = 0; i < line_count; i++) {
        fprintf(out_file, "%s\n", all_results[i]);
    }
    fclose(out_file);

    // Write performance metrics
    char* perf_filename = generate_performance_filename(config, config->input_file);
    FILE* perf_file = fopen(perf_filename, "a");
    if (!perf_file) fail("Could not open performance file for writing.");

    // Check if file is empty (new file) to write header
    fseek(perf_file, 0, SEEK_END);
    long file_size = ftell(perf_file);
    if (file_size == 0) {
        // File is empty, write header
        fprintf(perf_file, "threads,throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency_ms\n");
    }
    
    fprintf(perf_file, "%d,%.2f,%.2f,%.2f,%.4f\n",
            config->num_threads,
            throughput_input_per_sec,
            throughput_mbytes_per_sec,
            throughput_match_per_sec,
            latency_ms);
    fclose(perf_file);
    
    printf("Results written to '%s' and '%s'\n\n", output_filename, perf_filename);
    free(output_filename);
    free(perf_filename);

    // --- 8. Cleanup ---
    hs_free_database(database);
    for (long i = 0; i < pattern_count; i++) free(patterns[i]);
    free(patterns);
    free(ids);
    free(flags);
    for (long i = 0; i < line_count; i++) {
        free(lines[i]);
        free(all_results[i]);
    }
    free(lines);
    free(line_lengths);
    free(all_results);
    free(threads);
    free(thread_data);

    return EXIT_SUCCESS;
}


// --- GPU Mode Implementation ---

int run_gpu_mode(const config_t* config) {
    printf("Starting GPU mode processing...\n");
    
    // --- 1. Initialize CUDA ---
    int device_count;
    checkCudaError(cudaGetDeviceCount(&device_count), "Getting device count");
    if (device_count == 0) {
        fail("No CUDA-capable devices found");
    }
    
    cudaDeviceProp device_prop;
    checkCudaError(cudaGetDeviceProperties(&device_prop, 0), "Getting device properties");
    printf("Using GPU: %s\n", device_prop.name);
    
    // --- 2. Read and prepare patterns ---
    printf("Reading regex patterns from '%s'...\n", config->rules_file);
    long pattern_count = 0;
    long ignored_total_bytes;
    unsigned int* ignored_lengths;
    char** patterns = read_lines_from_file(config->rules_file, &pattern_count, &ignored_lengths, &ignored_total_bytes);
    free(ignored_lengths);
    printf("Loaded %ld patterns.\n", pattern_count);
    
    // --- 3. Read input data ---
    printf("Reading input data from '%s'...\n", config->input_file);
    long line_count = 0;
    long total_bytes = 0;
    unsigned int* line_lengths;
    char** lines = read_lines_from_file(config->input_file, &line_count, &line_lengths, &total_bytes);
    printf("Read %ld lines, total size: %.2f MB.\n", line_count, (double)total_bytes / (1024 * 1024));
    
    // --- 4. Create CUDA events for precise timing ---
    cudaEvent_t start_event, end_event;
    checkCudaError(cudaEventCreate(&start_event), "Creating start event");
    checkCudaError(cudaEventCreate(&end_event), "Creating end event");
    
    // Start timing (including data transfer)
    checkCudaError(cudaEventRecord(start_event, 0), "Recording start event");
    
    // --- 5. Allocate GPU memory ---
    printf("Allocating GPU memory...\n");
    
    // Allocate memory for line pointers and data
    char** d_lines;
    unsigned int* d_line_lengths;
    char** d_patterns;
    unsigned int* d_pattern_lengths;
    int* d_results;
    int* d_match_counts;
    
    checkCudaError(cudaMalloc(&d_lines, line_count * sizeof(char*)), "Allocating d_lines");
    checkCudaError(cudaMalloc(&d_line_lengths, line_count * sizeof(unsigned int)), "Allocating d_line_lengths");
    checkCudaError(cudaMalloc(&d_patterns, pattern_count * sizeof(char*)), "Allocating d_patterns");
    checkCudaError(cudaMalloc(&d_pattern_lengths, pattern_count * sizeof(unsigned int)), "Allocating d_pattern_lengths");
    checkCudaError(cudaMalloc(&d_results, line_count * pattern_count * sizeof(int)), "Allocating d_results");
    checkCudaError(cudaMalloc(&d_match_counts, line_count * sizeof(int)), "Allocating d_match_counts");
    
    // Allocate memory for actual string data
    char** d_line_data = (char**)malloc(line_count * sizeof(char*));
    char** d_pattern_data = (char**)malloc(pattern_count * sizeof(char*));
    unsigned int* h_pattern_lengths = (unsigned int*)malloc(pattern_count * sizeof(unsigned int));
    
    // Copy lines to GPU
    for (long i = 0; i < line_count; i++) {
        checkCudaError(cudaMalloc(&d_line_data[i], (line_lengths[i] + 1) * sizeof(char)), "Allocating line data");
        checkCudaError(cudaMemcpy(d_line_data[i], lines[i], (line_lengths[i] + 1) * sizeof(char), cudaMemcpyHostToDevice), "Copying line data");
    }
    
    // Copy patterns to GPU
    for (long i = 0; i < pattern_count; i++) {
        h_pattern_lengths[i] = strlen(patterns[i]);
        size_t pattern_len = h_pattern_lengths[i] + 1;
        checkCudaError(cudaMalloc(&d_pattern_data[i], pattern_len * sizeof(char)), "Allocating pattern data");
        checkCudaError(cudaMemcpy(d_pattern_data[i], patterns[i], pattern_len * sizeof(char), cudaMemcpyHostToDevice), "Copying pattern data");
    }
    
    // Copy pointer arrays to GPU
    checkCudaError(cudaMemcpy(d_lines, d_line_data, line_count * sizeof(char*), cudaMemcpyHostToDevice), "Copying line pointers");
    checkCudaError(cudaMemcpy(d_line_lengths, line_lengths, line_count * sizeof(unsigned int), cudaMemcpyHostToDevice), "Copying line lengths");
    checkCudaError(cudaMemcpy(d_patterns, d_pattern_data, pattern_count * sizeof(char*), cudaMemcpyHostToDevice), "Copying pattern pointers");
    checkCudaError(cudaMemcpy(d_pattern_lengths, h_pattern_lengths, pattern_count * sizeof(unsigned int), cudaMemcpyHostToDevice), "Copying pattern lengths");
    
    // --- 6. Launch GPU kernel ---
    printf("Launching GPU kernel...\n");
    int block_size = 256;
    int grid_size = (line_count + block_size - 1) / block_size;
    
    gpu_regex_match_kernel<<<grid_size, block_size>>>(d_lines, d_line_lengths, d_patterns, d_pattern_lengths,
                                                       (int)pattern_count, d_results, d_match_counts, (int)line_count);
    
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");
    
    // --- 7. Copy results back to host ---
    printf("Copying results back to host...\n");
    int* h_results = (int*)malloc(line_count * pattern_count * sizeof(int));
    int* h_match_counts = (int*)malloc(line_count * sizeof(int));
    
    checkCudaError(cudaMemcpy(h_results, d_results, line_count * pattern_count * sizeof(int), cudaMemcpyDeviceToHost), "Copying results");
    checkCudaError(cudaMemcpy(h_match_counts, d_match_counts, line_count * sizeof(int), cudaMemcpyDeviceToHost), "Copying match counts");
    
    // Stop timing
    checkCudaError(cudaEventRecord(end_event, 0), "Recording end event");
    checkCudaError(cudaEventSynchronize(end_event), "Synchronizing end event");
    
    // Calculate elapsed time
    float elapsed_ms;
    checkCudaError(cudaEventElapsedTime(&elapsed_ms, start_event, end_event), "Calculating elapsed time");
    double elapsed_seconds = elapsed_ms / 1000.0;
    
    // --- 8. Process results and calculate metrics ---
    printf("Processing results...\n");
    long total_matches = 0;
    char** all_results = (char**)malloc(line_count * sizeof(char*));
    
    for (long i = 0; i < line_count; i++) {
        int match_count = h_match_counts[i];
        total_matches += match_count;
        
        if (match_count > 0) {
            // Build result string with comma-separated pattern IDs
            size_t buffer_size = match_count * 10;
            char* result_buffer = (char*)malloc(buffer_size);
            int offset = 0;
            
            for (int j = 0; j < match_count; j++) {
                int pattern_id = h_results[i * pattern_count + j];
                offset += snprintf(result_buffer + offset, buffer_size - offset,
                                   "%d%s", pattern_id, (j == match_count - 1) ? "" : ",");
            }
            all_results[i] = result_buffer;
        } else {
            all_results[i] = strdup("");
        }
    }
    
    printf("GPU processing completed.\n");
    
    // --- 9. Calculate performance metrics ---
    double throughput_input_per_sec = line_count / elapsed_seconds;
    double throughput_mbytes_per_sec = (total_bytes / (1024.0 * 1024.0)) / elapsed_seconds;
    double throughput_match_per_sec = total_matches / elapsed_seconds;
    double latency_ms = (elapsed_seconds * 1000.0) / line_count;
    
    printf("Performance Metrics:\n");
    printf("  Total Time: %.4f seconds\n", elapsed_seconds);
    printf("  Total Matches: %ld\n", total_matches);
    printf("  Throughput (Input/sec): %.2f\n", throughput_input_per_sec);
    printf("  Throughput (MBytes/sec): %.2f\n", throughput_mbytes_per_sec);
    printf("  Throughput (Match/sec): %.2f\n", throughput_match_per_sec);
    printf("  Latency (ms/input): %.4f\n", latency_ms);
    
    // --- 10. Write output files ---
    char* output_filename = generate_output_filename(config);
    printf("Writing results to '%s'...\n", output_filename);
    
    // Write match results
    FILE* out_file = fopen(output_filename, "w");
    if (!out_file) fail("Could not open output file for writing.");
    for (long i = 0; i < line_count; i++) {
        fprintf(out_file, "%s\n", all_results[i]);
    }
    fclose(out_file);
    
    // Write performance metrics
    char* perf_filename = generate_performance_filename(config, config->input_file);
    FILE* perf_file = fopen(perf_filename, "a");
    if (!perf_file) fail("Could not open performance file for writing.");
    
    // Check if file is empty (new file) to write header
    fseek(perf_file, 0, SEEK_END);
    long file_size = ftell(perf_file);
    if (file_size == 0) {
        // File is empty, write header for GPU mode
        fprintf(perf_file, "matcher_name,throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency_ms\n");
    }
    
    fprintf(perf_file, "CUDA-SimpleMatcher,%.2f,%.2f,%.2f,%.4f\n",
            throughput_input_per_sec,
            throughput_mbytes_per_sec,
            throughput_match_per_sec,
            latency_ms);
    fclose(perf_file);
    
    printf("Results written to '%s' and '%s'\n\n", output_filename, perf_filename);
    
    // --- 11. Cleanup ---
    // Free GPU memory
    for (long i = 0; i < line_count; i++) {
        cudaFree(d_line_data[i]);
    }
    for (long i = 0; i < pattern_count; i++) {
        cudaFree(d_pattern_data[i]);
    }
    cudaFree(d_lines);
    cudaFree(d_line_lengths);
    cudaFree(d_patterns);
    cudaFree(d_pattern_lengths);
    cudaFree(d_results);
    cudaFree(d_match_counts);
    
    // Free host memory
    free(d_line_data);
    free(d_pattern_data);
    free(h_pattern_lengths);
    free(h_results);
    free(h_match_counts);
    
    for (long i = 0; i < pattern_count; i++) free(patterns[i]);
    free(patterns);
    for (long i = 0; i < line_count; i++) {
        free(lines[i]);
        free(all_results[i]);
    }
    free(lines);
    free(line_lengths);
    free(all_results);
    free(output_filename);
    free(perf_filename);
    
    // Cleanup CUDA events
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);
    
    return EXIT_SUCCESS;
}


// --- Main Function ---

int main(int argc, char* argv[]) {
    config_t config = parse_arguments(argc, argv);
    
    printf("High-Performance Regex Matching - Mode: %s\n", 
           config.mode == MODE_CPU ? "CPU" : "GPU");
    
    if (config.mode == MODE_CPU) {
        return run_cpu_mode(&config);
    } else {
        return run_gpu_mode(&config);
    }
}
