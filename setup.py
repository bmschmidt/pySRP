import os
from setuptools import setup

setup(name='SRP',
      packages=["SRP"],
      version='2.0.0',
      description="Run stable random projections.",
      long_description = open("README.rst").readlines(),
      url="http://github.com/bmschmidt/SRP",
      author="Benjamin Schmidt",
      author_email="bmschmidt@gmail.com",
      license="MIT",
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        "Natural Language :: English",
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        "Operating System :: Unix",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',          
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',          
        "Topic :: Text Processing :: Indexing",
        "Topic :: Text Processing :: Linguistic"
        ],
      install_requires= ["numpy", "regex"]
)
