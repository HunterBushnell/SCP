"""Install the existing source tree under its collision-free public name."""
from setuptools import find_packages, setup

setup(
    name="scp-pipeline",
    version="0.3.0",
    description="Single-cell characterization and tuning pipeline for NEURON",
    url="https://github.com/cyneuro/SCP",
    maintainer="Neural Engineering Laboratory at the University of Missouri",
    maintainer_email="hrbncv@missouri.edu",
    project_urls={
        "Source": "https://github.com/cyneuro/SCP",
        "Documentation": "https://github.com/cyneuro/SCP/blob/main/docs/README.md",
        "Issues": "https://github.com/cyneuro/SCP/issues",
    },
    python_requires=">=3.11",
    packages=["scp_pipeline", *["scp_pipeline." + p for p in find_packages("modules")]],
    package_dir={"scp_pipeline": "modules"},
    package_data={"scp_pipeline.analysis": ["*.json", "analysis_presets/*.json"]},
    entry_points={"console_scripts": ["scp-pipeline=scp_pipeline.cli:main"]},
    extras_require={
        "bmtool": ["bmtool==0.8.6"],
        "act": ["act @ git+https://github.com/V-Marco/ACT.git@2b3fdc190107b8242e3496329cfdf240abd09694"],
    },
    # The supported scientific environment is managed by environment.yml.
    # Installing the interface must not silently replace its pinned libraries.
    include_package_data=False,
)
