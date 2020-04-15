#!/usr/bin/env python3

import setuptools
import numpy as np 

ext = setuptools.Extension('horner', sources=['fem/horner.c'], include_dirs=[np.get_include()])

setuptools.setup(
	name='hotransport', 
	author='Samuel Olivier', 
	description='high order finite transport methods in 1D', 
	packages=['fem', 'transport'], 
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent"
		],
	python_requires=">=3.6", 
	install_requires=['numpy', 'scipy', 'termcolor', 'pyamg', 'matplotlib', 'quadpy', 'pathlib'], 
	ext_modules=[ext]
	)
