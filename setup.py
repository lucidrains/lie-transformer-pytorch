from setuptools import setup, find_packages

setup(
  name = 'lie-transformer-pytorch',
  packages = find_packages(),
  version = '0.0.16',
  license='MIT',
  description = 'Lie Transformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/lie-transformer-pytorch',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'equivariance',
    'lifting',
    'lie groups'
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.3'
  ],
  setup_requires=[
    'pytest-runner',
  ],
  tests_require=[
    'pytest'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
