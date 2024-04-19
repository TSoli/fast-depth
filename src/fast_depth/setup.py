from glob import glob

from setuptools import find_packages, setup

package_name = "fast_depth"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("lib/" + package_name, glob(f"{package_name}/*.py")),
        (f"lib/{package_name}/imagenet", glob(f"{package_name}/imagenet/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TSoli",
    maintainer_email="tariqsoliman2000@gmail.com",
    description="TODO: Package description",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fast_depth = fast_depth.node:main",
            "viewer = fast_depth.viewer:main",
        ],
    },
)
