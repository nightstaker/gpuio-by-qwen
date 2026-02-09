"""
Setup script for gpuio Python bindings.
"""

from setuptools import setup, Extension, find_packages
import os

# Get the directory containing this file
here = os.path.abspath(os.path.dirname(__file__))

# Define the extension module
gpuio_extension = Extension(
    'gpuio.gpuio',
    sources=[
        'src/python/gpuio_python.c',
        'src/gpuio.c',
    ],
    include_dirs=[
        'include',
        'include/gpuio',
    ],
    libraries=['cuda', 'rdmacm', 'ibverbs'],  # Optional: link if available
    library_dirs=[],
    define_macros=[],
    extra_compile_args=['-O3', '-fPIC'],
    language='c',
)

# Read README
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gpuio',
    version='1.0.0',
    description='GPU-Initiated IO Accelerator for AI/ML',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='gpuio Team',
    author_email='team@gpuio.ai',
    url='https://github.com/nightstaker/gpuio',
    
    packages=find_packages(where='src/python'),
    package_dir={'': 'src/python'},
    
    ext_modules=[gpuio_extension],
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Hardware',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: C',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    
    keywords='gpu io accelerator ai ml training inference deepseek',
    
    python_requires='>=3.8',
    
    install_requires=[
        'numpy>=1.20.0',
    ],
    
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'flake8',
        ],
        'torch': [
            'torch>=1.10.0',
        ],
    },
    
    entry_points={
        'console_scripts': [
            'gpuio-benchmark=gpuio.benchmark:main',
            'gpuio-monitor=gpuio.monitor:main',
        ],
    },
    
    project_urls={
        'Bug Reports': 'https://github.com/nightstaker/gpuio/issues',
        'Source': 'https://github.com/nightstaker/gpuio',
        'Documentation': 'https://gpuio.readthedocs.io',
    },
)
