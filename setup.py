from setuptools import setup, find_packages

VERSION = '0.1.17'
DESCRIPTION = 'A cutting-edge defect prediction tool with up-to-date Just-in-Time techniques and a robust API, empowering software development teams to proactively identify and mitigate defects in real-time'

# Setting up
setup(
    name="defectguard",
    version=VERSION,
    author="manhtd",
    author_email="manh.td194616@sis.hust.edu.vn",
    description=DESCRIPTION,
    packages=find_packages(),
    package_data={
        "defectguard.utils": ["*.json"],
    },
    install_requires=[
        'numpy',
        'pandas',
        'PyGithub',
        'Requests',
        'tqdm',
        'scikit-learn',
        'torch',
        'imblearn',
        'scipy',
        'gdown'
        ],
    keywords=['python', 'defect', 'prediction', 'just-in-time', 'defect prediction'],
    entry_points={
        'console_scripts': ['defectguard=defectguard:main'],
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)