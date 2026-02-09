/**
 * @file gpuio_python.c
 * @brief Python bindings for gpuio using Python C API
 * @version 1.0.0
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "gpuio.h"
#include "gpuio_ai.h"

/* Module definition */
static PyModuleDef gpuio_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "gpuio",
    .m_doc = "GPU-Initiated IO Accelerator for AI/ML",
    .m_size = -1,
};

/* Error object */
static PyObject* GPUIOError;

/* Convert gpuio_error_t to Python exception */
static void raise_gpuio_error(gpuio_error_t err) {
    PyErr_SetString(GPUIOError, gpuio_error_string(err));
}

/* ============================================================================
 * Context Object
 * ============================================================================ */

typedef struct {
    PyObject_HEAD
    gpuio_context_t ctx;
} ContextObject;

static void context_dealloc(ContextObject* self) {
    if (self->ctx) {
        gpuio_finalize(self->ctx);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* context_new(PyTypeObject* type, PyObject* args, 
                              PyObject* kwds) {
    ContextObject* self = (ContextObject*)type->tp_alloc(type, 0);
    if (!self) return NULL;
    self->ctx = NULL;
    return (PyObject*)self;
}

static int context_init(ContextObject* self, PyObject* args, PyObject* kwds) {
    gpuio_config_t config = GPUIO_CONFIG_DEFAULT;
    
    /* Parse optional config dict */
    PyObject* config_dict = NULL;
    if (!PyArg_ParseTuple(args, "|O", &config_dict)) {
        return -1;
    }
    
    if (config_dict && PyDict_Check(config_dict)) {
        /* Extract config values */
        PyObject* log_level = PyDict_GetItemString(config_dict, "log_level");
        if (log_level && PyLong_Check(log_level)) {
            config.log_level = (int)PyLong_AsLong(log_level);
        }
    }
    
    gpuio_error_t err = gpuio_init(&self->ctx, &config);
    if (err != GPUIO_SUCCESS) {
        raise_gpuio_error(err);
        return -1;
    }
    
    return 0;
}

static PyObject* context_get_device_count(ContextObject* self) {
    int count;
    gpuio_error_t err = gpuio_get_device_count(self->ctx, &count);
    if (err != GPUIO_SUCCESS) {
        raise_gpuio_error(err);
        return NULL;
    }
    return PyLong_FromLong(count);
}

static PyObject* context_get_stats(ContextObject* self) {
    gpuio_stats_t stats;
    gpuio_error_t err = gpuio_get_stats(self->ctx, &stats);
    if (err != GPUIO_SUCCESS) {
        raise_gpuio_error(err);
        return NULL;
    }
    
    PyObject* result = PyDict_New();
    PyDict_SetItemString(result, "requests_submitted",
                         PyLong_FromUnsignedLongLong(stats.requests_submitted));
    PyDict_SetItemString(result, "requests_completed",
                         PyLong_FromUnsignedLongLong(stats.requests_completed));
    PyDict_SetItemString(result, "bandwidth_gbps",
                         PyFloat_FromDouble(stats.bandwidth_gbps));
    PyDict_SetItemString(result, "cache_hit_rate",
                         PyFloat_FromDouble(stats.cache_hit_rate));
    
    return result;
}

static PyMethodDef context_methods[] = {
    {"get_device_count", (PyCFunction)context_get_device_count, METH_NOARGS,
     "Get number of available GPU devices"},
    {"get_stats", (PyCFunction)context_get_stats, METH_NOARGS,
     "Get IO statistics"},
    {NULL}
};

static PyTypeObject ContextType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "gpuio.Context",
    .tp_doc = "GPUIO Context",
    .tp_basicsize = sizeof(ContextObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = context_new,
    .tp_init = (initproc)context_init,
    .tp_dealloc = (destructor)context_dealloc,
    .tp_methods = context_methods,
};

/* ============================================================================
 * Module Initialization
 * ============================================================================ */

PyMODINIT_FUNC PyInit_gpuio(void) {
    PyObject* module = PyModule_Create(&gpuio_module);
    if (!module) return NULL;
    
    /* Add Context type */
    if (PyType_Ready(&ContextType) < 0) return NULL;
    Py_INCREF(&ContextType);
    PyModule_AddObject(module, "Context", (PyObject*)&ContextType);
    
    /* Add exception */
    GPUIOError = PyErr_NewException("gpuio.GPUIOError", NULL, NULL);
    Py_XINCREF(GPUIOError);
    PyModule_AddObject(module, "GPUIOError", GPUIOError);
    
    /* Add version */
    PyModule_AddStringConstant(module, "__version__", 
                               gpuio_get_version_string());
    
    /* Add constants */
    PyModule_AddIntConstant(module, "LOG_NONE", GPUIO_LOG_NONE);
    PyModule_AddIntConstant(module, "LOG_FATAL", GPUIO_LOG_FATAL);
    PyModule_AddIntConstant(module, "LOG_ERROR", GPUIO_LOG_ERROR);
    PyModule_AddIntConstant(module, "LOG_WARN", GPUIO_LOG_WARN);
    PyModule_AddIntConstant(module, "LOG_INFO", GPUIO_LOG_INFO);
    PyModule_AddIntConstant(module, "LOG_DEBUG", GPUIO_LOG_DEBUG);
    
    return module;
}
