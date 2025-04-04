from setuptools import setup, find_packages

setup(
    name='generative-models-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project implementing generative models including Deep CNN, Transformer, and Diffusion Model.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torchvision',
        'transformers',
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'PyYAML',
        'jupyter'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)