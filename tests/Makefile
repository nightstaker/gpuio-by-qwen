# Makefile for gpuio test suite

# Compiler settings
CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -I../include -I../include/gpuio
LDFLAGS = -L../lib -lgpuio -lpthread -lm

# Directories
UNIT_DIR = unit
INTEGRATION_DIR = integration
BENCHMARK_DIR = benchmark

# Test executables
UNIT_TESTS = test_core test_ai
INTEGRATION_TESTS = test_training test_inference
BENCHMARKS = bench_throughput

# All targets
.PHONY: all clean test unit integration benchmark

all: unit integration benchmark

# Unit tests
unit: $(UNIT_TESTS)

test_core: $(UNIT_DIR)/test_core.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_ai: $(UNIT_DIR)/test_ai.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Integration tests
integration: $(INTEGRATION_TESTS)

test_training: $(INTEGRATION_DIR)/test_training.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_inference: $(INTEGRATION_DIR)/test_inference.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Benchmarks
benchmark: $(BENCHMARKS)

bench_throughput: $(BENCHMARK_DIR)/bench_throughput.c
	$(CC) $(CFLAGS) -O3 -o $@ $< $(LDFLAGS)

# Run tests
test: all
	@echo "Running unit tests..."
	@./test_core || exit 1
	@./test_ai || exit 1
	@echo ""
	@echo "Running integration tests..."
	@./test_training || exit 1
	@./test_inference || exit 1
	@echo ""
	@echo "All tests passed!"

# Run benchmarks
run-benchmarks: benchmark
	@echo "Running benchmarks..."
	@./bench_throughput

# Clean
clean:
	rm -f $(UNIT_TESTS) $(INTEGRATION_TESTS) $(BENCHMARKS)
	rm -f *.o

# Help
help:
	@echo "gpuio Test Suite Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all          - Build all tests and benchmarks"
	@echo "  unit         - Build unit tests only"
	@echo "  integration  - Build integration tests only"
	@echo "  benchmark    - Build benchmarks only"
	@echo "  test         - Build and run all tests"
	@echo "  run-benchmarks - Build and run benchmarks"
	@echo "  clean        - Remove all built files"
	@echo "  help         - Show this help message"
