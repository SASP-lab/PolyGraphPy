from setuptools import setup, find_packages

setup(
    name='polygraphpy',
    version='0.1.0',
    author='João Gabriel Duarte',
    author_email='dacos033@umn.edu',
    description='A package for polymer graph-based property prediction and generative design.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SASP-lab/PolyGraphPy',
    packages=find_packages(),
    include_package_data=True, # This is crucial for including your data files
    install_requires=[
        'torch>=2.8.0',
        'torch-geometric>=2.6.1',
        'torch-cluster>=1.6.3',
        'torch-scatter>=2.1.2',
        'torch-sparse>=0.6.18',
        'torch-spline-conv>=1.2.2',
        'scikit-learn>=1.7.1',
        'selfies>=2.2.0',
        'stk>=2025.7.17.0',
        'transformers>=4.55.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
    python_requires='>=3.13',
)