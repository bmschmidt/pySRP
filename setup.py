from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='pysrp',
      packages=["SRP"],
      version='1.1.0',
      description="Run stable random projections.",
      long_description = long_description,
      long_description_content_type = "text/markdown",
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
        'Programming Language :: Python :: 3.9',
        "Topic :: Text Processing :: Indexing",
        "Topic :: Text Processing :: Linguistic"
        ],
      install_requires= ["numpy", "regex", "sqlitedict"]
)
