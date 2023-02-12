from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

use_cuda = True
if torch.cuda.is_available() and use_cuda:
    print('Including CUDA code.')
    setup(
        name='MiePy',
        ext_modules=[
            CUDAExtension('MiePy', [
                'src/MiePy_cuda.cpp',
                'src/MiePy_kernel.cu',
            ])
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
else:
    print('NO CUDA is found!')
    
