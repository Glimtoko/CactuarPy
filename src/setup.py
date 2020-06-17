# python setup.py build_ext --inplace

from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "core",
        ["core.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='core',
    ext_modules=cythonize(ext_modules, language_level="3", annotate=True),
)
