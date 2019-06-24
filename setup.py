import os
from setuptools import setup

setup(name='SRP',
      packages=["SRP"],
      version='1.1.0',
      description="Run stable random projections.",
      long_description = open("README.rst").readlines(),
      url="http://github.com/bmschmidt/SRP",
      author="Benjamin Schmidt",
      author_email="b.schmidt@neu.edu",
      license="MIT",
      # Copy the cgi-executable to a cgi-dir.
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        "Natural Language :: English",
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        "Operating System :: Unix",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.1',
        "Topic :: Text Processing :: Indexing",
        "Topic :: Text Processing :: Linguistic"
        ],
      install_requires= ["numpy", "regex", "future"]
)
