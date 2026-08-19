"""Build the optional E3 Cython solver kernel in place."""
from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy


setup(
    name="global-solver-kernel",
    ext_modules=cythonize(
        [Extension("global_solver_kernel", ["global_solver_kernel.pyx"], include_dirs=[numpy.get_include()])],
        compiler_directives={"language_level": "3"},
    ),
)
