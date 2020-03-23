from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gen_normal_mask',
    ext_modules=[
        CUDAExtension('gen_normal_mask', [
            'gen_normal_mask.cpp',
            'gen_normal_mask_cuda.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
