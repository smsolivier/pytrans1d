#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h> 

static PyObject* _polyval(PyObject* self, PyObject* args) {
	PyArrayObject *B; 
	double x; 

	if (!PyArg_ParseTuple(args, "Od", &B, &x)) {
		printf("oof\n"); 
		return NULL; 
	}

	int N = PyArray_DIM(B, 1); 
	int p = PyArray_DIM(B, 0); 

	double *ptr = PyArray_DATA(B); 
	npy_intp dims = {N}; 
	double* shape = malloc(sizeof(double)*N); 
	for (int i=0; i<N; i++) {
		shape[i] = ptr[i]; 
		for (int j=1; j<p; j++) {
			shape[i] = shape[i]*x + ptr[j*N+ i]; 
		}
	}

	return PyArray_SimpleNewFromData(1, &dims, NPY_DOUBLE, shape); 
}

static PyMethodDef mainMethods[] = {
	{"PolyVal", _polyval, METH_VARARGS, "horner's method"}, {NULL, NULL, 0, NULL}
}; 

static PyModuleDef pv_mod = {PyModuleDef_HEAD_INIT, "horner", "module for Horner's method", -1, mainMethods}; 

PyMODINIT_FUNC PyInit_horner(void) {
	import_array(); 
	return PyModule_Create(&pv_mod); 
}