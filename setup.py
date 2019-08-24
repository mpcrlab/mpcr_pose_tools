import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mpcr_pose_tools",
    version="0.0.1",
    author="Paul Morris",
    author_email="pmorris2012@fau.edu",
    description="Tools to preprocess and analyze pose data for FAU's MPCR lab.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mpcrlab/mpcr_pose_tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: Linux",
    ],
)