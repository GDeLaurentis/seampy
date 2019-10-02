from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='seampy',
    version='v1.0.0',
    license='GNU General Public License v3.0',
    description='Scattering Equations AMplitudes with PYthon',
    author='Giuseppe De Laurentis',
    author_email='g.dl@hotmail.it',
    url='https://github.com/GDeLaurentis/seampy',
    download_url='https://github.com/GDeLaurentis/seampy/archive/.tar.gz',
    keywords=['Scattering Equations', 'Numerical Amplitudes', 'Tree Level', 'CHY Formalism'],
    packages=find_packages(),
    install_requires=['numpy<1.17',
                      'mpmath<=1.1.0',
                      'sympy<=1.4',
                      'lips'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2.7',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
