from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setup(
    name="topsis-102003083-Utkarsh",
    version="1.1.0",
    description="A Python package implementing TOPSIS technique.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/UTK21/topsis-project",
    author="Utkarsh Chaudhary",
    author_email="ukguy21@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["topsis"],
    include_package_data=True,
    install_requires=[
                      'numpy',
                      'pandas'
     ],
    entry_points={
        "console_scripts": [
            "topsis=topsis.topsis:main",
        ]
    },
)