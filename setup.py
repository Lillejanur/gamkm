from setuptools import setup, find_packages
# Setting up
setup( 
    name="gamkm", 
    version="12.4.0",
    author="Mikael Valter Lithander and Minttu Kauppinen",
    author_email="<mikval@dtu.dk>",
    #url="https://github.com/avishart/gaussianprocess",
    description="Gaussian Process regression with stable hyperparameter optimization",
    long_description="Gaussian Process regression with stable hyperparameter optimization",
    packages=find_packages(),
    install_requires=['numpy>=1.22.3','scipy>=1.4.0','ase>=3.22.1','chemparse','func_timeout'], 
    #extras_require={'optional':['mpi4py>=3.0.3']},
    #test_suite='tests',
    #tests_require=['unittest'],
    keywords=['python','genetic algorithm','microkinetic model','adsorbate-adsorbate interaction'],
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.10'
)
