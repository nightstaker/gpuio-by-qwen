# gpuio API

[![CI](https://github.com/nightstaker/gpuio/actions/workflows/ci.yml/badge.svg)](https://github.com/nightstaker/gpuio/actions/workflows/ci.yml)
[![Build](https://github.com/nightstaker/gpuio/actions/workflows/build.yml/badge.svg)](https://github.com/nightstaker/gpuio/actions/workflows/build.yml)
[![CodeQL](https://github.com/nightstaker/gpuio/actions/workflows/codeql.yml/badge.svg)](https://github.com/nightstaker/gpuio/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GPU-Initiated IO Accelerator API for AI/ML Workloads

## Overview

This directory contains the C/C++ API headers and Python bindings for gpuio - a high-performance GPU-initiated IO acceleration framework designed for AI training and inference.

## Directory Structure

```
include/gpuio/       # C/C++ API headers
  gpuio.h             # Core API
  gpuio_ai.h          # AI/ML extensions (DSA, Engram, Graph RAG)

src/                  # Implementation
  gpuio.c             # Core implementation stub
  python/             # Python bindings
    gpuio_python.c    # C extension module
    gpuio/__init__.py # Python package

setup.py              # Python package setup
```

## C API Quick Start

```c
#include <gpuio/gpuio.h>
#include <gpuio/gpuio_ai.h>

// Initialize context
gpuio_context_t ctx;
gpuio_config_t config = GPUIO_CONFIG_DEFAULT;
gpuio_init(&ctx, &config);

// Create AI context
gpuio_ai_config_t ai_config = {
    .num_layers = 96,
    .num_heads = 128,
    .head_dim = 128,
    .enable_dsa_kv = true,
    .enable_engram = true,
};
gpuio_ai_context_t ai_ctx;
gpuio_ai_context_create(ctx, &ai_config, &ai_ctx);

// Create DSA KV cache pool
gpuio_dsa_kv_config_t kv_config = {
    .hbm_capacity = 10ULL * 1024 * 1024 * 1024,  // 10GB
    .cxl_capacity = 100ULL * 1024 * 1024 * 1024, // 100GB
    .default_compression = GPUIO_KV_COMPRESS_FP16,
};
gpuio_dsa_kv_pool_t kv_pool;
gpuio_dsa_kv_pool_create(ai_ctx, &kv_config, &kv_pool);

// Load KV cache entry
gpuio_dsa_kv_entry_t entry;
gpuio_dsa_kv_load(kv_pool, position, layer_id, head_id, stream, &entry);

// Cleanup
gpuio_dsa_kv_pool_destroy(kv_pool);
gpuio_ai_context_destroy(ai_ctx);
gpuio_finalize(ctx);
```

## Python API Quick Start

```python
import gpuio

# Initialize context
ctx = gpuio.Context({"log_level": gpuio.LOG_INFO})

# Get device info
num_gpus = ctx.get_device_count()
print(f"Available GPUs: {num_gpus}")

# Check stats
stats = ctx.get_stats()
print(f"Bandwidth: {stats['bandwidth_gbps']:.2f} GB/s")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

## Key Features

### Core API (`gpuio.h`)
- **Context Management**: Initialize and configure gpuio
- **Device Management**: Query GPU capabilities
- **Memory Management**: Allocate and register memory for zero-copy
- **Request Management**: Submit async/sync IO requests
- **On-Demand Proxy**: Transparent data access with lazy loading

### AI Extensions (`gpuio_ai.h`)

#### DeepSeek DSA KV Cache
- Sparse attention pattern management
- Tiered storage (HBM/CXL/Remote)
- Compression (FP16/INT8/4BIT/Sparse)
- Importance-based eviction

```c
gpuio_dsa_kv_load(pool, position, layer, head, stream, &entry);
gpuio_dsa_kv_store(pool, position, layer, head, data, size, importance, stream);
```

#### Graph RAG
- Vector similarity scatter for node retrieval
- Multi-hop subgraph gather
- GPU-accelerated ANN index

```c
gpuio_graph_rag_scatter(index, &scatter_params, stream, &candidates, &n);
gpuio_graph_rag_gather(storage, candidates, n, &gather_params, stream, &subgraph, &size);
```

#### DeepSeek Engram Memory
- Petabyte-scale external memory
- Learned content addressing
- Semantic query with embeddings

```c
gpuio_engram_write(pool, &engram, stream);
gpuio_engram_query(pool, query_embedding, threshold, top_k, stream, results, &n);
```

## Building

### C Library

```bash
# Compile library
gcc -c -O3 -fPIC -I include src/gpuio.c -o libgpuio.o
ar rcs libgpuio.a libgpuio.o

# Or shared library
gcc -shared -o libgpuio.so libgpuio.o
```

### Python Package

```bash
# Install dependencies
pip install numpy

# Build and install
python setup.py build_ext --inplace
pip install -e .

# Or with pip
pip install .
```

## API Reference

See the header files for complete API documentation:
- `include/gpuio/gpuio.h` - Core API
- `include/gpuio/gpuio_ai.h` - AI/ML extensions

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core API | Stub | Interface defined, implementation pending |
| DSA KV Cache | Stub | Data structures defined |
| Graph RAG | Stub | Interface defined |
| Engram Memory | Stub | Interface defined |
| Python Bindings | Stub | Basic structure implemented |

## License

MIT License - See LICENSE file
