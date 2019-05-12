from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# run the command: python lda_build.py build_ext --inplace
setup(
    cmdclass     = {'build_ext': build_ext},
    include_dirs = [np.get_include()], 
    ext_modules  = [
                   Extension("lda_sample_z_ids", ["lda_sample_z_ids.pyx"], extra_compile_args = ['-fopenmp', '-O3'], extra_link_args = ['-fopenmp'], language = 'c++')
                   ]
)

