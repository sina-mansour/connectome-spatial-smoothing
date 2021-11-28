import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requirements = ["numpy", "scipy", "sklearn", "nibabel", "gdist"]

setuptools.setup(
    name="Connectome_Spatial_Smoothing",
    version="0.1.1",
    author="Sina Mansour L.",
    author_email="sina.mansour.lakouraj@gmail.com",
    description="Connectome Spatial Smoothing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sina-mansour/connectome-spatial-smoothing",
    project_urls={
        "Bug Tracker": "https://github.com/sina-mansour/connectome-spatial-smoothing/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "code"},
    packages=setuptools.find_packages(where="code"),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
