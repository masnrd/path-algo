from setuptools import setup, find_packages

setup(
    name='pathfinder',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'h3',
        'folium',
        'scipy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
