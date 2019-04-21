import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ido_cv",
    version="0.0.1",
    author="Ilya Dobrynin, Fedor Pesyak",
    author_email="iliadobrynin@yandex.ru, fedoroffworking@gmail.com",
    description="Package for computer vision tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)