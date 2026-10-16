// SPDX-License-Identifier: MIT
// Embedded Python for the JAX material paradigm. Buffers cross as memoryviews
// on the Fortran arrays (no copy, no numpy C API); python/fe_jax.py reads and
// writes them in Fortran order. Every call checks for a Python exception.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <time.h>

#define MAXHANDLES 64

static PyObject *module = NULL;
static PyObject *handles[MAXHANDLES];
static int nhandles = 0;

// Start Python as the interpreter python_exe (a venv's python picks up its
// site-packages) and import fe_jax from module_dir. Returns 0 on success.
int pyb_init(const char *python_exe, const char *module_dir) {
  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  PyStatus status = PyConfig_SetBytesString(&config, &config.executable, python_exe);
  if (!PyStatus_Exception(status)) status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status)) {
    fprintf(stderr, "pyb_init: %s\n", status.err_msg ? status.err_msg : "Python failed to start");
    return -1;
  }
  PyObject *path = PySys_GetObject("path");  // borrowed
  PyObject *dir = PyUnicode_DecodeFSDefault(module_dir);
  if (!path || !dir || PyList_Insert(path, 0, dir) < 0) {
    Py_XDECREF(dir);
    PyErr_Print();
    return -1;
  }
  Py_DECREF(dir);
  module = PyImport_ImportModule("fe_jax");
  if (!module) {
    PyErr_Print();
    return -1;
  }
  return 0;
}

// Handle of fe_jax.make(model, cfg), or -1.
int pyb_handle(const char *model, const char *cfg) {
  if (nhandles == MAXHANDLES) return -1;
  PyObject *h = PyObject_CallMethod(module, "make", "ss", model, cfg);
  if (!h) {
    PyErr_Print();
    return -1;
  }
  handles[nhandles] = h;
  return nhandles++;
}

// stress(ns, n), tangent(nt, n) from f(3, 3, n) and par(npar). Returns 0 on success.
int pyb_eval(int h, int n, double *f, double *par, int npar, double *stress, int ns,
             double *tangent, int nt) {
  PyObject *args[4] = {
      PyMemoryView_FromMemory((char *)f, (Py_ssize_t)9 * n * sizeof(double), PyBUF_READ),
      PyMemoryView_FromMemory((char *)par, (Py_ssize_t)npar * sizeof(double), PyBUF_READ),
      PyMemoryView_FromMemory((char *)stress, (Py_ssize_t)ns * n * sizeof(double), PyBUF_WRITE),
      PyMemoryView_FromMemory((char *)tangent, (Py_ssize_t)nt * n * sizeof(double), PyBUF_WRITE)};
  PyObject *r = NULL;
  if (args[0] && args[1] && args[2] && args[3])
    r = PyObject_CallFunctionObjArgs(handles[h], args[0], args[1], args[2], args[3], NULL);
  for (int i = 0; i < 4; i++) Py_XDECREF(args[i]);
  if (!r) {
    PyErr_Print();
    return -1;
  }
  Py_DECREF(r);
  return 0;
}

void pyb_finalize(void) {
  for (int i = 0; i < nhandles; i++) Py_DECREF(handles[i]);
  nhandles = 0;
  Py_XDECREF(module);
  module = NULL;
  Py_FinalizeEx();
}

// Monotonic wall time in seconds: JAX runs threads, so CPU time would overcount.
double pyb_wtime(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
