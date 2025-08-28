# High-Performance Regex Matching using Hyperscan

This project implements a high-performance regex matching system using Intel's Hyperscan library, supporting both CPU multi-threading and GPU processing modes.

## Features

- **CPU Mode**: Multi-threaded processing using pthreads
- **GPU Mode**: CUDA-based processing (implementation in progress)
- **Hyperscan Integration**: Uses Intel's Hyperscan library for optimized pattern matching
- **Performance Metrics**: Detailed throughput and latency measurements
- **Automatic Output**: Results and performance data saved automatically

## Dataset Files

The large dataset files are compressed to reduce repository size:
- `set1.tar.gz` - Compressed version of set1.txt
- `set2.tar.gz` - Compressed version of set2.txt  
- `set3.tar.gz` - Compressed version of set3.txt

### Extracting Dataset Files

Before running the program, extract the dataset files:

```bash
tar -xzf set1.tar.gz
tar -xzf set2.tar.gz
tar -xzf set3.tar.gz
```

Or extract all at once:
```bash
tar -xzf set*.tar.gz
```

## Build Instructions

1. Install Hyperscan library:
   ```bash
   sudo apt-get install libhyperscan-dev  # Ubuntu/Debian
   # or
   brew install hyperscan  # macOS
   ```

2. Build the project:
   ```bash
   make
   ```

## Usage

### CPU Mode
```bash
./bin/HW3_MCC_030402_401106039 --mode cpu --rules rules.txt --input set1.txt --threads 4
```

### GPU Mode (Coming Soon)
```bash
./bin/HW3_MCC_030402_401106039 --mode gpu --rules rules.txt --input set1.txt
```

## Output Files

Results are automatically saved in the `results/` directory:
- Match results: `Results_HW3_MCC_030402_401106039_{CPU/GPU}_{DataSet}_{NumThreads/Library}.txt`
- Performance metrics: `Results_HW3_MCC_030402_401106039_{CPU/GPU}_{DataSet}_{Library}.csv`

## Performance Metrics

The program measures:
- **Throughput**: Input lines per second, MBytes per second, Matches per second
- **Latency**: Processing time per input line
- **Total Execution Time**: Complete processing duration

## Project Structure

```
├── src/
│   └── main.c              # Main implementation
├── bin/                    # Compiled executables
├── results/                # Output files
├── rules.txt              # Regex patterns
├── set*.tar.gz            # Compressed dataset files
├── Makefile               # Build configuration
└── README.md              # This file
```
