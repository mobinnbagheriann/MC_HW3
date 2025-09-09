#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <hs/hs.h>

#define BUFFER_SIZE_LIMIT 100000
#define PATTERN_ARRAY_SIZE 100000

typedef struct {
    char **rule_expressions;
    unsigned int *rule_identifiers;
    unsigned int *compilation_flags;
    int total_rules;
    hs_database_t *compiled_database;
} pattern_storage_t;

typedef struct {
    char **text_lines;
    int *line_byte_counts;
    int line_total;
    int start_index;
    int end_index;
    int worker_id;
    int worker_count;
    pattern_storage_t *pattern_data;
    char **output_results;
    int *match_tallies;
    double *execution_times;
    long long bytes_handled;
    int matches_found;
} worker_context_t;

typedef struct {
    int *detected_rules;
    int detection_count;
    int storage_capacity;
    int *rule_tracker; // Array to track which rules have been found (for uniqueness)
    int max_rule_count;   // Maximum number of rules to track
} detection_state_t;

// Global variables for performance metrics
pthread_mutex_t statistics_lock = PTHREAD_MUTEX_INITIALIZER;
double cumulative_execution_time = 0.0;
long long cumulative_data_volume = 0;
int cumulative_detections = 0;
int cumulative_processed_lines = 0;

// Utility function for numerical sorting
int numerical_comparator(const void *first, const void *second) {
    int value_a = *(const int*)first;
    int value_b = *(const int*)second;
    return (value_a > value_b) - (value_a < value_b);
}

// Pattern detection callback function
static int pattern_detection_callback(unsigned int id, unsigned long long from, 
                        unsigned long long to, unsigned int flags, void *context) {
    (void)from;   // Suppress unused parameter warning
    (void)to;     // Suppress unused parameter warning  
    (void)flags;  // Suppress unused parameter warning
    
    detection_state_t *state = (detection_state_t *)context;
    
    // Check if this rule has already been found for this line
    int rule_number = id + 1; // Convert to 1-based indexing
    if (rule_number <= state->max_rule_count && !state->rule_tracker[rule_number - 1]) {
        // Mark this rule as found
        state->rule_tracker[rule_number - 1] = 1;
        
        // Resize array if needed
        if (state->detection_count >= state->storage_capacity) {
            state->storage_capacity *= 2;
            state->detected_rules = realloc(state->detected_rules, state->storage_capacity * sizeof(int));
            if (!state->detected_rules) {
                fprintf(stderr, "Failed to reallocate memory for matches\n");
                return 1;
            }
        }
        
        state->detected_rules[state->detection_count++] = rule_number;
    }
    
    return 0; // Continue matching
}

// Memory allocation for pattern storage
static int allocate_pattern_storage(pattern_storage_t *storage) {
    storage->rule_expressions = malloc(PATTERN_ARRAY_SIZE * sizeof(char*));
    storage->rule_identifiers = malloc(PATTERN_ARRAY_SIZE * sizeof(unsigned int));
    storage->compilation_flags = malloc(PATTERN_ARRAY_SIZE * sizeof(unsigned int));
    storage->total_rules = 0;
    
    if (!storage->rule_expressions || !storage->rule_identifiers || !storage->compilation_flags) {
        return -1;
    }
    return 0;
}

// Read patterns from file and populate storage
static int read_pattern_file(const char *filename, pattern_storage_t *storage) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open rules file");
        return -1;
    }
    
    char line[BUFFER_SIZE_LIMIT];
    while (fgets(line, sizeof(line), file) && storage->total_rules < PATTERN_ARRAY_SIZE) {
        // Remove newline
        line[strcspn(line, "\r\n")] = 0;
        if (strlen(line) > 0) {
            storage->rule_expressions[storage->total_rules] = strdup(line);
            storage->rule_identifiers[storage->total_rules] = storage->total_rules;
            storage->compilation_flags[storage->total_rules] = HS_FLAG_DOTALL | HS_FLAG_MULTILINE;
            storage->total_rules++;
        }
    }
    
    fclose(file);
    return 0;
}

// Compile patterns into hyperscan database
static int compile_pattern_database(pattern_storage_t *storage) {
    hs_compile_error_t *compile_err;
    if (hs_compile_multi((const char **)storage->rule_expressions,
                        storage->compilation_flags,
                        storage->rule_identifiers,
                        storage->total_rules,
                        HS_MODE_BLOCK,
                        NULL,
                        &storage->compiled_database,
                        &compile_err) != HS_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to compile patterns: %s\n", compile_err->message);
        hs_free_compile_error(compile_err);
        return -1;
    }
    return 0;
}

// Combined function to initialize pattern storage
int initialize_pattern_engine(const char *filename, pattern_storage_t *storage) {
    if (allocate_pattern_storage(storage) != 0) {
        return -1;
    }
    
    if (read_pattern_file(filename, storage) != 0) {
        return -1;
    }
    
    if (compile_pattern_database(storage) != 0) {
        return -1;
    }
    
    printf("Loaded and compiled %d regex patterns\n", storage->total_rules);
    return 0;
}

// Count total lines in input file
static int count_file_lines(FILE *file) {
    char buffer[BUFFER_SIZE_LIMIT];
    int line_count = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        line_count++;
    }
    return line_count;
}

// Process individual line and clean formatting
static int process_input_line(char *buffer, char **processed_line, int *processed_length) {
    int len = strlen(buffer);
    if (len > 0 && (buffer[len-1] == '\n' || buffer[len-1] == '\r')) {
        buffer[len-1] = '\0';
        if (len > 1 && buffer[len-2] == '\r') {
            buffer[len-2] = '\0';
            len--;
        }
        len--;
    }
    
    *processed_line = strdup(buffer);
    *processed_length = len;
    return (*processed_line != NULL) ? 0 : -1;
}

// Function to load input data into memory
int parse_input_data(const char *filename, char ***lines, int **line_lengths, int *total_lines) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open input file");
        return -1;
    }
    
    // Count lines first
    *total_lines = count_file_lines(file);
    
    // Allocate memory for lines
    *lines = malloc(*total_lines * sizeof(char*));
    *line_lengths = malloc(*total_lines * sizeof(int));
    
    // Read and process lines
    rewind(file);
    char buffer[BUFFER_SIZE_LIMIT];
    int line_index = 0;
    while (fgets(buffer, sizeof(buffer), file) && line_index < *total_lines) {
        if (process_input_line(buffer, &(*lines)[line_index], &(*line_lengths)[line_index]) != 0) {
            fclose(file);
            return -1;
        }
        line_index++;
    }
    
    fclose(file);
    printf("Loaded %d lines from input file\n", *total_lines);
    return 0;
}

// Initialize detection state for a single line
static void initialize_detection_state(detection_state_t *state, int max_rules) {
    state->detected_rules = malloc(100 * sizeof(int));
    state->detection_count = 0;
    state->storage_capacity = 100;
    state->max_rule_count = max_rules;
    state->rule_tracker = calloc(max_rules, sizeof(int));
}

// Build comma-separated result string from detections
static char* build_result_string(detection_state_t *state) {
    if (state->detection_count > 0) {
        char *result = malloc(state->detection_count * 10 + 1);
        result[0] = '\0';
        
        for (int j = 0; j < state->detection_count; j++) {
            char temp[20];
            sprintf(temp, "%d", state->detected_rules[j]);
            strcat(result, temp);
            if (j < state->detection_count - 1) {
                strcat(result, ",");
            }
        }
        return result;
    } else {
        return strdup("");
    }
}

// Process single line for pattern matches
static int process_line_patterns(worker_context_t *context, int line_index, hs_scratch_t *scratch) {
    detection_state_t detection_state;
    initialize_detection_state(&detection_state, context->pattern_data->total_rules);
    
    // Search for matches
    if (hs_scan(context->pattern_data->compiled_database,
               context->text_lines[line_index],
               context->line_byte_counts[line_index],
               0,
               scratch,
               pattern_detection_callback,
               &detection_state) != HS_SUCCESS) {
        fprintf(stderr, "Worker %d: Scan failed for line %d\n", context->worker_id, line_index);
        free(detection_state.detected_rules);
        free(detection_state.rule_tracker);
        return -1;
    }
    
    // Sort the matched rule numbers in ascending order
    if (detection_state.detection_count > 0) {
        qsort(detection_state.detected_rules, detection_state.detection_count, sizeof(int), numerical_comparator);
    }
    
    // Build and store result
    context->output_results[line_index] = build_result_string(&detection_state);
    
    int matches_this_line = detection_state.detection_count;
    free(detection_state.detected_rules);
    free(detection_state.rule_tracker);
    
    return matches_this_line;
}

// Worker thread execution function
void* execute_pattern_matching(void* arg) {
    worker_context_t *context = (worker_context_t*)arg;
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    // Allocate scratch space for this thread
    hs_scratch_t *scratch = NULL;
    if (hs_alloc_scratch(context->pattern_data->compiled_database, &scratch) != HS_SUCCESS) {
        fprintf(stderr, "Worker %d: Failed to allocate scratch space\n", context->worker_id);
        return NULL;
    }
    
    long long worker_bytes = 0;
    int worker_matches = 0;
    
    // Process assigned lines
    for (int i = context->start_index; i < context->end_index; i++) {
        int line_matches = process_line_patterns(context, i, scratch);
        if (line_matches >= 0) {
            worker_bytes += context->line_byte_counts[i];
            worker_matches += line_matches;
        }
    }
    
    gettimeofday(&end_time, NULL);
    double worker_time = (end_time.tv_sec - start_time.tv_sec) + 
                        (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    
    // Update global metrics
    pthread_mutex_lock(&statistics_lock);
    cumulative_execution_time += worker_time;
    cumulative_data_volume += worker_bytes;
    cumulative_detections += worker_matches;
    cumulative_processed_lines += (context->end_index - context->start_index);
    pthread_mutex_unlock(&statistics_lock);
    
    context->bytes_handled = worker_bytes;
    context->matches_found = worker_matches;
    context->execution_times[context->worker_id] = worker_time;
    
    hs_free_scratch(scratch);
    return NULL;
}

// Function to save results to output file
int save_output_data(const char *filename, char **results, int total_lines) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Failed to create output file");
        return -1;
    }
    
    for (int i = 0; i < total_lines; i++) {
        fprintf(file, "%s\n", results[i]);
    }
    
    fclose(file);
    return 0;
}

// Extract base filename from full path
static void extract_base_filename(const char *input_filename, char *output_buffer, size_t buffer_size) {
    const char *base_name = strrchr(input_filename, '/');
    if (!base_name) {
        base_name = strrchr(input_filename, '\\');
    }
    if (base_name) {
        base_name++; // Skip the slash
    } else {
        base_name = input_filename;
    }
    
    strncpy(output_buffer, base_name, buffer_size - 1);
    output_buffer[buffer_size - 1] = '\0';
    
    char *dot = strrchr(output_buffer, '.');
    if (dot) {
        *dot = '\0';
    }
}

// Calculate performance metrics
static void calculate_performance_metrics(int total_lines, double total_time, long long total_bytes, 
                                        int total_matches, double *throughput_input, 
                                        double *throughput_mbytes, double *throughput_matches, 
                                        double *latency) {
    *throughput_input = total_lines / total_time;
    *throughput_mbytes = (total_bytes / (1024.0 * 1024.0)) / total_time;
    *throughput_matches = total_matches / total_time;
    *latency = (total_time * 1000.0) / total_lines; // in milliseconds
}

// Function to export performance statistics to CSV
int export_performance_statistics(const char *input_filename, int num_threads, double total_time, 
                     long long total_bytes, int total_matches, int total_lines) {
    // Extract filename without path and extension
    char filename_without_ext[256];
    extract_base_filename(input_filename, filename_without_ext, sizeof(filename_without_ext));
    
    // Create CSV filename
    char csv_filename[512];
    snprintf(csv_filename, sizeof(csv_filename), "performance_metric_%s.csv", filename_without_ext);
    
    // Check if file exists
    FILE *test_file = fopen(csv_filename, "r");
    int file_exists = (test_file != NULL);
    if (test_file) {
        fclose(test_file);
    }
    
    // Open file for appending
    FILE *file = fopen(csv_filename, "a");
    if (!file) {
        perror("Failed to create/open CSV file");
        return -1;
    }
    
    // Write header only if file didn't exist
    if (!file_exists) {
        fprintf(file, "threads,throughput_input_per_sec,throughput_mbytes_per_sec,throughput_match_per_sec,latency\n");
    }
    
    double throughput_input, throughput_mbytes, throughput_matches, latency;
    calculate_performance_metrics(total_lines, total_time, total_bytes, total_matches,
                                &throughput_input, &throughput_mbytes, &throughput_matches, &latency);
    
    fprintf(file, "%d,%.2f,%.2f,%.2f,%.4f\n", 
            num_threads, throughput_input, throughput_mbytes, throughput_matches, latency);
    
    fclose(file);
    
    printf("Performance metrics appended to: %s\n", csv_filename);
    return 0;
}

// Allocate and initialize worker contexts
static int setup_worker_contexts(worker_context_t **contexts, pthread_t **threads, 
                                double **times, int num_threads, int total_lines,
                                char **input_lines, int *line_lengths, 
                                pattern_storage_t *pattern_data, char **results) {
    *threads = malloc(num_threads * sizeof(pthread_t));
    *contexts = malloc(num_threads * sizeof(worker_context_t));
    *times = malloc(num_threads * sizeof(double));
    
    if (!*threads || !*contexts || !*times) {
        return -1;
    }
    
    int lines_per_worker = total_lines / num_threads;
    int remaining_lines = total_lines % num_threads;
    int current_line = 0;
    
    for (int i = 0; i < num_threads; i++) {
        (*contexts)[i].text_lines = input_lines;
        (*contexts)[i].line_byte_counts = line_lengths;
        (*contexts)[i].line_total = total_lines;
        (*contexts)[i].start_index = current_line;
        (*contexts)[i].end_index = current_line + lines_per_worker + (i < remaining_lines ? 1 : 0);
        (*contexts)[i].worker_id = i;
        (*contexts)[i].worker_count = num_threads;
        (*contexts)[i].pattern_data = pattern_data;
        (*contexts)[i].output_results = results;
        (*contexts)[i].execution_times = *times;
        
        current_line = (*contexts)[i].end_index;
    }
    
    return 0;
}

// Launch worker threads and wait for completion
static int execute_parallel_processing(pthread_t *threads, worker_context_t *contexts, int num_threads) {
    // Create and start threads
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&threads[i], NULL, execute_pattern_matching, &contexts[i]) != 0) {
            perror("Failed to create thread");
            return -1;
        }
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    return 0;
}

// Display performance summary
static void display_performance_summary(int num_threads, int total_lines, double total_wall_time) {
    printf("Processing completed in %.2f seconds\n", total_wall_time);
    printf("Total matches found: %d\n", cumulative_detections);
    printf("Total bytes processed: %lld\n", cumulative_data_volume);
    
    printf("\nPerformance Metrics:\n");
    printf("Threads: %d\n", num_threads);
    printf("Throughput (Input/sec): %.2f lines/sec\n", total_lines / total_wall_time);
    printf("Throughput (MBytes/sec): %.2f MB/sec\n", (cumulative_data_volume / (1024.0 * 1024.0)) / total_wall_time);
    printf("Throughput (Match/sec): %.2f matches/sec\n", cumulative_detections / total_wall_time);
    printf("Latency (ms): %.4f ms/line\n", (total_wall_time * 1000.0) / total_lines);
}

// Cleanup allocated resources
static void cleanup_resources(char **input_lines, char **results, int total_lines,
                             pthread_t *threads, worker_context_t *contexts, double *times,
                             pattern_storage_t *pattern_data, int *line_lengths) {
    for (int i = 0; i < total_lines; i++) {
        free(input_lines[i]);
        free(results[i]);
    }
    free(input_lines);
    free(line_lengths);
    free(results);
    free(threads);
    free(contexts);
    free(times);
    
    for (int i = 0; i < pattern_data->total_rules; i++) {
        free(pattern_data->rule_expressions[i]);
    }
    free(pattern_data->rule_expressions);
    free(pattern_data->rule_identifiers);
    free(pattern_data->compilation_flags);
    
    hs_free_database(pattern_data->compiled_database);
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <rules_file> <input_file> <output_file> <num_threads>\n", argv[0]);
        return 1;
    }
    
    const char *rules_file = argv[1];
    const char *input_file = argv[2];
    const char *output_file = argv[3];
    int num_threads = atoi(argv[4]);
    
    if (num_threads <= 0) {
        fprintf(stderr, "Number of threads must be positive\n");
        return 1;
    }
    
    printf("Starting Hyperscan matcher with %d threads\n", num_threads);
    
    // Initialize pattern engine
    pattern_storage_t pattern_data;
    if (initialize_pattern_engine(rules_file, &pattern_data) != 0) {
        return 1;
    }
    
    // Load input data
    char **input_lines;
    int *line_lengths;
    int total_lines;
    if (parse_input_data(input_file, &input_lines, &line_lengths, &total_lines) != 0) {
        return 1;
    }
    
    // Allocate result arrays
    char **results = malloc(total_lines * sizeof(char*));
    for (int i = 0; i < total_lines; i++) {
        results[i] = NULL;
    }
    
    // Setup worker contexts
    pthread_t *threads;
    worker_context_t *worker_contexts;
    double *execution_times;
    if (setup_worker_contexts(&worker_contexts, &threads, &execution_times, num_threads, 
                             total_lines, input_lines, line_lengths, &pattern_data, results) != 0) {
        return 1;
    }
    
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    // Execute parallel processing
    if (execute_parallel_processing(threads, worker_contexts, num_threads) != 0) {
        return 1;
    }
    
    gettimeofday(&end_time, NULL);
    double total_wall_time = (end_time.tv_sec - start_time.tv_sec) + 
                            (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    
    display_performance_summary(num_threads, total_lines, total_wall_time);
    
    // Save results
    if (save_output_data(output_file, results, total_lines) != 0) {
        return 1;
    }
    
    // Export performance metrics
    if (export_performance_statistics(input_file, num_threads, total_wall_time, 
                         cumulative_data_volume, cumulative_detections, total_lines) != 0) {
        return 1;
    }
    
    printf("Results written to: %s\n", output_file);
    
    // Cleanup resources
    cleanup_resources(input_lines, results, total_lines, threads, worker_contexts, 
                     execution_times, &pattern_data, line_lengths);
    
    return 0;
}
