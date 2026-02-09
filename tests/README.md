# gpuio Test Suite

Comprehensive test suite for gpuio covering unit tests, integration tests, and performance benchmarks.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_core.c         # Core API tests (context, memory, streams)
│   └── test_ai.c           # AI extension tests (DSA, Engram, Graph RAG)
├── integration/            # End-to-end integration tests
│   ├── test_training.c     # Training workload tests
│   └── test_inference.c    # Inference workload tests
├── benchmark/              # Performance benchmarks
│   └── bench_throughput.c  # Throughput and latency benchmarks
├── python/                 # Python binding tests
│   └── test_gpuio.py       # Python unit tests
└── CMakeLists.txt          # Build configuration
```

## Test Coverage

### Unit Tests (test_core.c)

**Context Management:**
- Context initialization with default and custom configs
- Double initialization prevention
- NULL argument handling

**Version Information:**
- Version string format
- Version number retrieval

**Error Handling:**
- Error code to string conversion
- Invalid error code handling

**Device Management:**
- Device count enumeration
- Device info retrieval
- Device selection
- Invalid device ID handling

**Memory Management:**
- Memory allocation and deallocation
- Zero-size allocation handling
- Large allocation attempts
- Memory registration for zero-copy
- NULL pointer handling

**Stream Management:**
- Stream creation and destruction
- Priority levels (default, high, low)
- Stream synchronization
- Stream query for idle status

**Event Management:**
- Event creation and destruction
- Event recording and synchronization

**Statistics:**
- Stats retrieval
- Stats reset

### AI Unit Tests (test_ai.c)

**AI Context:**
- AI context creation with full config
- Minimal config handling

**DSA KV Cache:**
- Pool creation and destruction
- Pool reset
- Single entry load/store
- Batch entry load
- Sparsity pattern configuration
- KV cache compaction
- Statistics retrieval

**Graph RAG:**
- Graph index creation/destruction
- Scatter parameter validation
- Gather parameter validation
- Graph storage backend

**Engram Memory:**
- Pool creation and destruction
- Single engram write/read
- Semantic query
- Batch write operations
- Synchronization
- Statistics retrieval

**Compression:**
- Codec creation and destruction
- Multiple codec types (LZ4, ZSTD, GZIP, FP16, INT8)

### Integration Tests

**Training Workloads (test_training.c):**
- Distributed training with checkpointing
- Large model checkpoint/restart (100GB)
- Multi-GPU data parallel simulation
- Fault tolerance with recovery

**Inference Workloads (test_inference.c):**
- Real-time inference latency (P99 target: <50ms)
- Graph RAG for knowledge-augmented LLM
- Engram-based long-context inference (100K tokens)
- Batched inference throughput

### Benchmarks (bench_throughput.c)

**MemIO Benchmarks:**
- Memory copy bandwidth at various sizes (4KB to 256MB)
- Throughput measurements in MB/s

**DSA KV Cache Benchmarks:**
- Single entry access latency
- Batch access throughput
- Cache hit rate measurement
- HBM/CXL tier statistics

**Engram Memory Benchmarks:**
- Query latency (target: <100μs)
- Query throughput (queries/sec)
- Cache hit rate

## Building Tests

### Using CMake

```bash
mkdir build
cd build
cmake ..
make
```

### Build Options

```bash
# Build only unit tests
cmake -DBUILD_UNIT_TESTS=ON -DBUILD_INTEGRATION_TESTS=OFF ..

# Build with coverage
cmake -DENABLE_COVERAGE=ON ..

# Build all (including benchmarks)
cmake -DBUILD_BENCHMARKS=ON ..
```

## Running Tests

### Run All Tests

```bash
# Using CMake
cmake --build build --target check

# Or using ctest
cd build && ctest --output-on-failure
```

### Run Individual Test Suites

```bash
# Unit tests
./build/test_core
./build/test_ai

# Integration tests
./build/test_training
./build/test_inference

# Benchmarks
./build/bench_throughput
```

### Python Tests

```bash
# Install gpuio Python package first
pip install ../..

# Run Python tests
python tests/python/test_gpuio.py

# Or using pytest
pytest tests/python/test_gpuio.py -v
```

## Test Requirements

### From System-Requirements.md

The test suite validates the following high-priority requirements:

| Use Case | Test File | Test Function |
|----------|-----------|---------------|
| UC-1.1.1: Deep Learning Training Checkpoint | test_training.c | test_distributed_training_with_checkpoint |
| UC-1.1.2: Real-time Inference Data Loading | test_inference.c | test_real_time_inference_latency |
| UC-1.2.1: Parallel Model Training with Prefetching | test_training.c | test_multi_gpu_data_parallel |
| UC-2.1.1: Out-of-Core GNN Training | test_ai.c | test_ai_context_create_destroy |
| UC-2.4.3: DeepSeek DSA KV Cache | test_ai.c | test_dsa_kv_* |
| UC-3.2.3: Graph RAG | test_ai.c | test_graph_rag_* |
| UC-4.2.3: DeepSeek Engram Memory | test_ai.c | test_engram_* |

### Performance Targets

| Metric | Target | Test |
|--------|--------|------|
| MemIO Bandwidth | >90% peak | bench_memio_memcpy |
| DSA KV Cache Hit Rate | >95% | test_dsa_kv_stats |
| DSA Access Latency | <5μs (cached) | bench_dsa_kv_access |
| Engram Query Latency | <100μs | bench_engram_query |
| Inference P99 Latency | <50ms | test_real_time_inference_latency |
| Checkpoint Throughput | >5GB/s | test_large_model_checkpoint_restart |

## Expected Output

### Unit Tests

```
gpuio Core API Unit Tests
Version: 1.0.0

Context Tests
============================================================
  Running context_init_default... PASSED
  Running context_init_custom_config... PASSED
  ...

============================================================
Test Summary:
  Total:  25
  Passed: 25
  Failed: 0
============================================================
```

### Integration Tests

```
============================================================
gpuio Training Workload Integration Tests
============================================================

Test: Distributed Training with Checkpointing
----------------------------------------------
  Simulating 3 epochs...
    Epoch 1/3: 123.4 ms
    Epoch 2/3: [CKPT] 145.6 ms
    Epoch 3/3: 122.1 ms

  Results:
    Total training time: 391.1 ms
    Checkpoints saved: 1
    Avg checkpoint time: 45.2 ms
    KV cache hit rate: 97.50%

  PASSED
```

### Benchmarks

```
============================================================
gpuio Performance Benchmarks
============================================================

Benchmark: MemIO - Memory Copy
-------------------------------
  Size:      4 MB | Bandwidth: 12345.67 MB/s | Latency: 0.32 us
  Size:     64 MB | Bandwidth: 11500.23 MB/s | Latency: 5.57 us
  ...

Benchmark: DSA KV Cache - Access Patterns
------------------------------------------
  Single Access:
    Min:  0.123 us
    Avg:  0.456 us
    P99:  1.234 us
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Install dependencies
        run: sudo apt-get install -y cmake build-essential
      
      - name: Build tests
        run: |
          mkdir build
          cd build
          cmake ..
          make
      
      - name: Run tests
        run: |
          cd build
          ctest --output-on-failure
      
      - name: Run benchmarks
        run: ./build/bench_throughput
```

## Adding New Tests

### Adding a Unit Test

1. Edit `tests/unit/test_core.c` or `tests/unit/test_ai.c`
2. Add test function:

```c
TEST(my_new_feature) {
    // Setup
    gpuio_context_t ctx;
    gpuio_init(&ctx, NULL);
    
    // Test
    gpuio_error_t err = my_new_function(ctx);
    ASSERT_EQ(err, GPUIO_SUCCESS);
    
    // Cleanup
    gpuio_finalize(ctx);
}
```

3. Add to test runner:

```c
RUN_TEST(my_new_feature);
```

### Adding an Integration Test

1. Edit `tests/integration/test_training.c` or `test_inference.c`
2. Add test function following the pattern of existing tests
3. Update the main function to call the new test

### Adding a Benchmark

1. Edit `tests/benchmark/bench_throughput.c`
2. Add benchmark function:

```c
static void bench_my_feature(void) {
    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        // Operation
    }
    
    // Benchmark
    double times[BENCHMARK_ITERATIONS];
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        double start = get_time_us();
        // Operation
        double end = get_time_us();
        times[i] = end - start;
    }
    
    // Report
    bench_stats_t stats;
    calculate_stats(times, BENCHMARK_ITERATIONS, &stats);
    print_stats("My Feature", &stats, "us");
}
```

## Troubleshooting

### Tests Fail to Build

- Ensure gpuio library is built first: `cd .. && make`
- Check CMake version: `cmake --version` (need 3.16+)
- Verify all dependencies are installed

### Tests Fail at Runtime

- Check that GPU drivers are installed
- Verify sufficient GPU memory is available
- Run with debug logging: `GPUIO_LOG_LEVEL=5 ./test_core`

### Benchmarks Show Low Performance

- Ensure tests are built with `-O3` optimization
- Check GPU is not being used by other processes
- Verify system is in performance mode (not power saving)

## License

MIT License - See LICENSE file
