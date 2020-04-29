#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h> 

static PyObject* _outer(PyObject* self, PyObject* args) {
	PyArrayObject *v1, *v2; 
	double alpha; 
	if (!PyArg_ParseTuple(args, "dOO", &alpha, &v1, &v2)) return NULL; 

	int m = PyArray_DIM(v1, 0); 
	int n = PyArray_DIM(v2, 0); 

	double *pv1 = PyArray_DATA(v1); 
	double *pv2 = PyArray_DATA(v2); 
	double* mat = malloc(sizeof(double)*m*n); 
	const npy_intp dims[2] = {m, n}; 
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			mat[j+i*n] = alpha*pv1[i] * pv2[j]; 
		}
	}
	return PyArray_SimpleNewFromData(2, &dims[0], NPY_DOUBLE, mat); 
}

static PyObject* _AddOuter(PyObject* self, PyObject* args) {
	PyArrayObject *v, *w, *elmat; 
	double alpha; 
	if (!PyArg_ParseTuple(args, "dOOO", &alpha, &v, &w, &elmat)) return NULL; 

	int m = PyArray_DIM(v, 0); 
	int n = PyArray_DIM(w, 0); 

	double *pv = PyArray_DATA(v); 
	double *pw = PyArray_DATA(w); 
	double *pm = PyArray_DATA(elmat); 
	const npy_intp dims[] = {m, n}; 
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			pm[j+i*n] += pv[i] * pw[j] * alpha; 
		}
	}
	Py_INCREF(Py_None); 
	return Py_None; 
}

static PyObject* _Mult(PyObject* self, PyObject* args) {
	PyArrayObject *A, *B; 
	double alpha; 
	if (!PyArg_ParseTuple(args, "dOO", &alpha, &A, &B)) return NULL; 

	int m = PyArray_DIM(A, 0); 
	int n = PyArray_DIM(A, 1); 
	int p = PyArray_DIM(B, 1); 

	double *a = PyArray_DATA(A); 
	double *b = PyArray_DATA(B); 
	double *c = malloc(sizeof(double)*m*p); 
	npy_intp dims[2] = {m,p}; 
	for (int i=0; i<m; i++) {
		for (int j=0; j<p; j++) {
			c[j+i*p] = 0.; 
			for (int k=0; k<n; k++) {
				c[j+i*p] += a[k+i*n]*b[j+k*p]; 
			}
			c[j+i*p] *= alpha; 
		}
	}
	return PyArray_SimpleNewFromData(2, &dims[0], NPY_DOUBLE, c); 
}

static PyObject* _AddMult(PyObject* self, PyObject* args) {
	PyArrayObject *A, *B, *C; 
	double alpha, beta; 
	if (!PyArg_ParseTuple(args, "dOOdO", &alpha, &A, &B, &beta, &C)) return NULL; 

	int m = PyArray_DIM(A, 0); 
	int n = PyArray_DIM(A, 1); 
	int p = PyArray_DIM(B, 1); 

	double *a = PyArray_DATA(A); 
	double *b = PyArray_DATA(B); 
	double *c = PyArray_DATA(C); 
	for (int i=0; i<m; i++) {
		for (int j=0; j<p; j++) {
			c[j+i*p] = beta*c[j+p*i]; 
			for (int k=0; k<n; k++) {
				c[j+p*i] += alpha*a[k+i*n]*b[j+k*p]; 
			}
		}
	}
	Py_INCREF(Py_None); 
	return Py_None; 
}

static PyObject* _TransMult(PyObject* self, PyObject* args) {
	PyArrayObject *A, *B; 
	if (!PyArg_ParseTuple(args, "OO", &A, &B)) return NULL; 

	int m = PyArray_DIM(A, 1); 
	int n = PyArray_DIM(A, 0); 
	int p = PyArray_DIM(B, 1); 

	double *a = PyArray_DATA(A); 
	double *b = PyArray_DATA(B); 
	double *mat = malloc(sizeof(double)*m*p); 
	npy_intp dims[2] = {m,p}; 
	for (int i=0; i<m; i++) {
		for (int j=0; j<p; j++) {
			mat[j+i*p] = 0; 
			for (int k=0; k<n; k++) {
				mat[j+p*i] += a[i+k*m]*b[j+k*p]; 
			}
		}
	}
	return PyArray_SimpleNewFromData(2, &dims[0], NPY_DOUBLE, mat); 
}

static PyObject* _AddTransMult(PyObject* self, PyObject* args) {
	PyArrayObject *A, *B, *C; 
	double alpha, beta; 
	if (!PyArg_ParseTuple(args, "dOOdO", &alpha, &A, &B, &beta, &C)) return NULL; 

	int m = PyArray_DIM(A, 1); 
	int n = PyArray_DIM(A, 0); 
	int p = PyArray_DIM(B, 1); 

	double *a = PyArray_DATA(A); 
	double *b = PyArray_DATA(B); 
	double *c = PyArray_DATA(C); 
	for (int i=0; i<m; i++) {
		for (int j=0; j<p; j++) {
			c[j+i*p] = beta*c[j+p*i]; 
			for (int k=0; k<n; k++) {
				c[j+p*i] += alpha*a[i+k*m]*b[j+k*p]; 
			}
		}
	}
	Py_INCREF(Py_None); 
	return Py_None; 
}

static PyMethodDef mainMethods[] = {
	{"Outer", _outer, METH_VARARGS, "outer product of two vectors"}, 
	{"AddOuter", _AddOuter, METH_VARARGS, "outer product of two row vectors multiplied by a constant"}, 
	{"Mult", _Mult, METH_VARARGS, "multiply two matrices"}, 
	{"AddMult", _AddMult, METH_VARARGS, "multiply accumulate matrices"}, 
	{"TransMult", _TransMult, METH_VARARGS, "transpose matrix multiply"}, 
	{"AddTransMult", _AddTransMult, METH_VARARGS, "matrix transpose multiply accumulate"}, 
	{NULL, NULL, 0, NULL}
}; 

static PyModuleDef mod = {PyModuleDef_HEAD_INIT, "linalg", 
	"module for c extensions that numpy should already have", -1, mainMethods}; 

PyMODINIT_FUNC PyInit_linalg(void) {
	import_array(); 
	return PyModule_Create(&mod); 
}