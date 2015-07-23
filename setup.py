from Cython.Distutils import build_ext
import cython_gsl

from setuptools import setup, find_packages, Extension

readme = open('README.rst').read()

setup(name='peircebayes',
      version='0.0.1',
      author='Calin-Rares Turliuc',
      author_email='ct1810@imperial.ac.uk',
      license='MIT',
      description='PeirceBayes - Probabilistic Logic Programming',
      long_description=readme,
      packages=find_packages(),
      package_data={
        'peircebayes': ['aprob.pl'],
      },
      install_requires = ['numpy>=1.9.2','scipy>=0.15.1','matplotlib>=1.4.3','cython>=0.22.1'],
      entry_points={
        'console_scripts': ['peircebayes = peircebayes.peircebayes:main']
      },
      include_dirs = [cython_gsl.get_include()],
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("peircebayes.prob_inference_dev",
                        ["peircebayes/prob_inference_dev.pyx"],
                        libraries=cython_gsl.get_libraries(),
                        library_dirs=[cython_gsl.get_library_dir()],
                        include_dirs=[cython_gsl.get_cython_include_dir()])]
      )
