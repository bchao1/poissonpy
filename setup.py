from setuptools import setup, find_packages

setup(
      name='poissonpy',
      version='1.0.0',
      description='Plug-and-play 2D Poisson equations library.',
      long_description="hi",
      url='https://github.com/bchao1/poissonpy',
      keywords = "pde poisson",
      author='bchao1',
      license='MIT',
      packages=find_packages(),
      python_requires=">=3.6"
)