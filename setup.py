import setuptools

version = exec(open("ldaflavours/version.py").read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ldaflavours",
    version=__version__,
    author="Ken Schröder",
    author_email="k.h.b.schroder@gmail.com",
    description="Toolbox for various labeled LDA problems, like Labeled LDA, HSLDA and Cascade LDA ",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KenHBS/LDA_Thesis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
