"""
gpuio - GPU-Initiated IO Accelerator for AI/ML

This package provides Python bindings for gpuio, a high-performance 
GPU-initiated IO acceleration framework designed for:
- Large-scale distributed AI training
- Real-time and batch inference serving
- DeepSeek DSA KV Cache Management
- Graph RAG for Knowledge-Augmented LLMs
- DeepSeek Engram Memory Architecture

Example:
    >>> import gpuio
    >>> 
    >>> # Initialize context
    >>> ctx = gpuio.Context({"log_level": gpuio.LOG_INFO})
    >>> 
    >>> # Get device count
    >>> print(f"GPUs: {ctx.get_device_count()}")
    >>> 
    >>> # Check stats
    >>> stats = ctx.get_stats()
    >>> print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
"""

from .gpuio import (
    Context,
    GPUIOError,
    LOG_NONE,
    LOG_FATAL,
    LOG_ERROR,
    LOG_WARN,
    LOG_INFO,
    LOG_DEBUG,
)

__version__ = "1.0.0"
__all__ = [
    "Context",
    "GPUIOError",
    "LOG_NONE",
    "LOG_FATAL",
    "LOG_ERROR",
    "LOG_WARN",
    "LOG_INFO",
    "LOG_DEBUG",
]
