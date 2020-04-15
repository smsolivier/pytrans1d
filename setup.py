#!/usr/bin/env python3

import setuptools

class get_numpy_include(object):
	def __str__(self):
		import numpy 
		return numpy.get_include()

ext = setuptools.Extension('trans1d.fem.horner', 
	sources=['trans1d/fem/horner.c'], 
	include_dirs=[get_numpy_include()], 
	extra_compile_args=['-fopenmp'], 
	extra_link_args=['-lgomp'])

setuptools.setup(
	name='trans1d', 
	author='Samuel Olivier', 
	description='high order finite transport methods in 1D', 
	packages=['trans1d', 'trans1d.fem', 'trans1d.transport'], 
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent"
		],
	python_requires=">=3.6", 
	install_requires=['numpy', 'scipy', 'termcolor', 'pyamg', 'matplotlib', 'quadpy', 'pathlib'], 
	ext_modules=[ext]
	)
