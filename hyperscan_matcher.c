#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <hs/hs.h>

#define MAX_LINE_LENGTH 100000
#define MAX_REGEX_COUNT 100000

typedef struct {
    char **patterns;
    unsigned int *ids;
    unsigned int *flags;
    int pattern_count;
    hs_database_t *database;
} regex_data_t;

typedef struct {
    char **input_lines;
    int *line_lengths;
    int total_lines;
    int start_line;
    int end_line;
    int thread_id;
    int num_threads;
    regex_data_t *regex_data;
    char **results;
    int *match_counts;
    double *processing_times;
    long long total_bytes_processed;
    int total_matches;
} thread_data_t;

typedef struct {
    int *matched_rules;
    int match_count;
    int capacity;
    int *rule_found; // Array to track which rules have been found (for uniqueness)
    int max_rules;   // Maximum number of rules to track
} match_context_t;

// Global variables for performance metrics
pthread_mutex_t metrics_mutex = PTHREAD_MUTEX_INITIALIZER;
double total_processing_time = 0.0;
long long total_bytes = 0;
int total_match_count = 0;
int total_lines_processed = 0;

// Comparison function for sorting rule numbers
int compare_ints(const void *a, const void *b) {
    int int_a = *(const int*)a;
    int int_b = *(const int*)b;
    return (int_a > int_b) - (int_a < int_b);
}

// Match callback function
static int match_handler(unsigned int id, unsigned long long from, 
                        unsigned long long to, unsigned int flags, void *context) {
    (void)from;   // Suppress unused parameter warning
    (void)to;     // Suppress unused parameter warning  
    (void)flags;  // Suppress unused parameter warning
    
    match_context_t *ctx = (match_context_t *)context;
    
    // Check if this rule has already been found for this line
    int rule_num = id + 1; // Convert to 1-based indexing
    if (rule_num <= ctx->max_rules && !ctx->rule_found[rule_num - 1]) {
        // Mark this rule as found
        ctx->rule_found[rule_num - 1] = 1;
        
        // Resize array if needed
        if (ctx->match_count >= ctx->capacity) {
            ctx->capacity *= 2;
            ctx->matched_rules = realloc(ctx->matched_rules, ctx->capacity * sizeof(int));
            if (!ctx->matched_rules) {
                fprintf(stderr, "Failed to reallocate memory for matches\n");
                return 1;
            }
        }
        
        ctx->matched_rules[ctx->match_count++] = rule_num;
    }
    
    return 0; // Continue matching
}

// Function to load regex patterns from file
int load_patterns(const char *filename, regex_data_t *regex_data) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open rules file");
        return -1;
    }
    
    char line[MAX_LINE_LENGTH];
    regex_data->patterns = malloc(MAX_REGEX_COUNT * sizeof(char*));
    regex_data->ids = malloc(MAX_REGEX_COUNT * sizeof(unsigned int));
    regex_data->flags = malloc(MAX_REGEX_COUNT * sizeof(unsigned int));
    regex_data->pattern_count = 0;
    
    while (fgets(line, sizeof(line), file) && regex_data->pattern_count < MAX_REGEX_COUNT) {
        // Remove newline
        line[strcspn(line, "\r\n")] = 0;
        if (strlen(line) > 0) {
            regex_data->patterns[regex_data->pattern_count] = strdup(line);
            regex_data->ids[regex_data->pattern_count] = regex_data->pattern_count;
            regex_data->flags[regex_data->pattern_count] = HS_FLAG_DOTALL | HS_FLAG_MULTILINE;
            regex_data->pattern_count++;
        }
    }
    
    fclose(file);
    
    // Compile Hyperscan database
    hs_compile_error_t *compile_err;
    if (hs_compile_multi((const char **)regex_data->patterns,
                        regex_data->flags,
                        regex_data->ids,
                        regex_data->pattern_count,
                        HS_MODE_BLOCK,
                        NULL,
                        &regex_data->database,
                        &compile_err) != HS_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to compile patterns: %s\n", compile_err->message);
        hs_free_compile_error(compile_err);
        return -1;
    }
    
    printf("Loaded and compiled %d regex patterns\n", regex_data->pattern_count);
    return 0;
}

// Function to load input file into memory
int load_input_file(const char *filename, char ***lines, int **line_lengths, int *total_lines) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Failed to open input file");
        return -1;
    }
    
    // Count lines first
    char buffer[MAX_LINE_LENGTH];
    *total_lines = 0;
    while (fgets(buffer, sizeof(buffer), file)) {
        (*total_lines)++;
    }
    
    // Allocate memory for lines
    *lines = malloc(*total_lines * sizeof(char*));
    *line_lengths = malloc(*total_lines * sizeof(int));
    
    // Read lines
    rewind(file);
    int line_count = 0;
    while (fgets(buffer, sizeof(buffer), file) && line_count < *total_lines) {
        // Remove newline but keep original length for metrics
        int len = strlen(buffer);
        if (len > 0 && (buffer[len-1] == '\n' || buffer[len-1] == '\r')) {
            buffer[len-1] = '\0';
            if (len > 1 && buffer[len-2] == '\r') {
                buffer[len-2] = '\0';
                len--;
            }
            len--;
        }
        
        (*lines)[line_count] = strdup(buffer);
        (*line_lengths)[line_count] = len;
        line_count++;
    }
    
    fclose(file);
    printf("Loaded %d lines from input file\n", *total_lines);
    return 0;
}

// Worker thread function
void* worker_thread(void* arg) {
    thread_data_t *data = (thread_data_t*)arg;
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    // Allocate scratch space for this thread
    hs_scratch_t *scratch = NULL;
    if (hs_alloc_scratch(data->regex_data->database, &scratch) != HS_SUCCESS) {
        fprintf(stderr, "Thread %d: Failed to allocate scratch space\n", data->thread_id);
        return NULL;
    }
    
    long long thread_bytes = 0;
    int thread_matches = 0;
    
    // Process assigned lines
    for (int i = data->start_line; i < data->end_line; i++) {
        match_context_t match_ctx;
        match_ctx.matched_rules = malloc(100 * sizeof(int)); // Initial capacity
        match_ctx.match_count = 0;
        match_ctx.capacity = 100;
        match_ctx.max_rules = data->regex_data->pattern_count;
        match_ctx.rule_found = calloc(match_ctx.max_rules, sizeof(int)); // Initialize to 0
        
        // Search for matches
        if (hs_scan(data->regex_data->database,
                   data->input_lines[i],
                   data->line_lengths[i],
                   0,
                   scratch,
                   match_handler,
                   &match_ctx) != HS_SUCCESS) {
            fprintf(stderr, "Thread %d: Scan failed for line %d\n", data->thread_id, i);
        }
        
        // Sort the matched rule numbers in ascending order
        if (match_ctx.match_count > 0) {
            qsort(match_ctx.matched_rules, match_ctx.match_count, sizeof(int), compare_ints);
        }
        
        // Build result string
        if (match_ctx.match_count > 0) {
            char *result = malloc(match_ctx.match_count * 10 + 1); // Rough estimate
            result[0] = '\0';
            
            for (int j = 0; j < match_ctx.match_count; j++) {
                char temp[20];
                sprintf(temp, "%d", match_ctx.matched_rules[j]);
                strcat(result, temp);
                if (j < match_ctx.match_count - 1) {
                    strcat(result, ",");
                }
            }
            data->results[i] = result;
        } else {
            data->results[i] = strdup(""); // Empty line for no matches
        }
        
        thread_bytes += data->line_lengths[i];
        thread_matches += match_ctx.match_count;
        
        free(match_ctx.matched_rules);
        free(match_ctx.rule_found);
    }
    
    gettimeofday(&end_time, NULL);
    double thread_time = (end_time.tv_sec - start_time.tv_sec) + 
                        (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    
    // Update global metrics
    pthread_mutex_lock(&metrics_mutex);
    total_processing_time += thread_time;
    total_bytes += thread_bytes;
    total_match_count += thread_matches;
    total_lines_processed += (data->end_line - data->start_line);
    pthread_mutex_unlock(&metrics_mutex);
    
    data->total_bytes_processed = thread_bytes;
    data->total_matches = thread_matches;
    data->processing_times[data->thread_id] = thread_time;
    
    hs_free_scratch(scratch);
    return NULL;
}

// Function to write results to file
int write_results(const char *filename, char **results, int total_lines) {
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

// Function to write performance metrics to CSV
int write_metrics_csv(const char *input_filename, int num_threads, double total_time, 
                     long long total_bytes, int total_matches, int total_lines) {
    // Extract input filename without path and extension
    const char *base_name = strrchr(input_filename, '/');
    if (!base_name) {
        base_name = strrchr(input_filename, '\\');
    }
    if (base_name) {
        base_name++; // Skip the slash
    } else {
        base_name = input_filename;
    }
    
    // Remove extension
    char filename_without_ext[256];
    strncpy(filename_without_ext, base_name, sizeof(filename_without_ext) - 1);
    filename_without_ext[sizeof(filename_without_ext) - 1] = '\0';
    
    char *dot = strrchr(filename_without_ext, '.');
    if (dot) {
        *dot = '\0';
    }
    
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
    
    double throughput_input = total_lines / total_time;
    double throughput_mbytes = (total_bytes / (1024.0 * 1024.0)) / total_time;
    double throughput_matches = total_matches / total_time;
    double latency = (total_time * 1000.0) / total_lines; // in milliseconds
    
    fprintf(file, "%d,%.2f,%.2f,%.2f,%.4f\n", 
            num_threads, throughput_input, throughput_mbytes, throughput_matches, latency);
    
    fclose(file);
    
    printf("Performance metrics appended to: %s\n", csv_filename);
    return 0;
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
    
    // Load regex patterns
    regex_data_t regex_data;
    if (load_patterns(rules_file, &regex_data) != 0) {
        return 1;
    }
    
    // Load input file
    char **input_lines;
    int *line_lengths;
    int total_lines;
    if (load_input_file(input_file, &input_lines, &line_lengths, &total_lines) != 0) {
        return 1;
    }
    
    // Allocate result arrays
    char **results = malloc(total_lines * sizeof(char*));
    for (int i = 0; i < total_lines; i++) {
        results[i] = NULL;
    }
    
    // Create thread data
    pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = malloc(num_threads * sizeof(thread_data_t));
    double *processing_times = malloc(num_threads * sizeof(double));
    
    int lines_per_thread = total_lines / num_threads;
    int remaining_lines = total_lines % num_threads;
    
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    // Create and start threads
    int current_line = 0;
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].input_lines = input_lines;
        thread_data[i].line_lengths = line_lengths;
        thread_data[i].total_lines = total_lines;
        thread_data[i].start_line = current_line;
        thread_data[i].end_line = current_line + lines_per_thread + (i < remaining_lines ? 1 : 0);
        thread_data[i].thread_id = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].regex_data = &regex_data;
        thread_data[i].results = results;
        thread_data[i].processing_times = processing_times;
        
        current_line = thread_data[i].end_line;
        
        if (pthread_create(&threads[i], NULL, worker_thread, &thread_data[i]) != 0) {
            perror("Failed to create thread");
            return 1;
        }
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    gettimeofday(&end_time, NULL);
    double total_wall_time = (end_time.tv_sec - start_time.tv_sec) + 
                            (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    
    printf("Processing completed in %.2f seconds\n", total_wall_time);
    printf("Total matches found: %d\n", total_match_count);
    printf("Total bytes processed: %lld\n", total_bytes);
    
    // Write results
    if (write_results(output_file, results, total_lines) != 0) {
        return 1;
    }
    
    // Write performance metrics
    if (write_metrics_csv(input_file, num_threads, total_wall_time, 
                         total_bytes, total_match_count, total_lines) != 0) {
        return 1;
    }
    
    printf("Results written to: %s\n", output_file);
    
    // Performance summary
    printf("\nPerformance Metrics:\n");
    printf("Threads: %d\n", num_threads);
    printf("Throughput (Input/sec): %.2f lines/sec\n", total_lines / total_wall_time);
    printf("Throughput (MBytes/sec): %.2f MB/sec\n", (total_bytes / (1024.0 * 1024.0)) / total_wall_time);
    printf("Throughput (Match/sec): %.2f matches/sec\n", total_match_count / total_wall_time);
    printf("Latency (ms): %.4f ms/line\n", (total_wall_time * 1000.0) / total_lines);
    
    // Cleanup
    for (int i = 0; i < total_lines; i++) {
        free(input_lines[i]);
        free(results[i]);
    }
    free(input_lines);
    free(line_lengths);
    free(results);
    free(threads);
    free(thread_data);
    free(processing_times);
    
    for (int i = 0; i < regex_data.pattern_count; i++) {
        free(regex_data.patterns[i]);
    }
    free(regex_data.patterns);
    free(regex_data.ids);
    free(regex_data.flags);
    
    hs_free_database(regex_data.database);
    
    return 0;
}
