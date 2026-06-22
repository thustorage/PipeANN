import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}

# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        # The Milvus-compatible gRPC server binary is bundled into the package
        # (pipeann/_bin/) so `pipeann-server` works straight after install.
        # CMAKE_RUNTIME_OUTPUT_DIRECTORY drops the executable there directly.
        bindir = extdir / "_bin"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={bindir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
            "-DBUILD_PYTHON_INTERFACE=ON",
            "-DBUILD_MILVUS_SERVER=ON",
            "-DUSE_TCMALLOC=OFF",
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            [
                "cmake", "--build", ".",
                "--target", ext.name.split(".")[-1],
                "--target", "pipeann_milvus_server",
                *build_args,
            ],
            cwd=build_temp,
            check=True,
        )

    def copy_extensions_to_source(self) -> None:
        # Called for in-place builds (`build_ext --inplace`, used by
        # `pip install -e .`). The base implementation mirrors each compiled
        # extension into the source tree; setuptools does not know about the
        # sibling _bin/ directory, so copy the bundled server binary too.
        super().copy_extensions_to_source()
        build_py = self.get_finalized_command("build_py")
        pkg_dir = Path(build_py.get_package_dir("pipeann"))
        src_bin = Path(self.build_lib) / "pipeann" / "_bin" / "pipeann_milvus_server"
        if src_bin.is_file():
            dst_dir = pkg_dir / "_bin"
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_bin, dst_dir / "pipeann_milvus_server")


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
setup(
    name="pipeann",
    version="0.4.0",
    author="Hao Guo",
    author_email="gh23@mails.tsinghua.edu.cn",
    description="Python wrapper for PipeANN",
    long_description="",
    # Tell setuptools to look for packages in the python directory
    packages=["pipeann"],
    # Tell setuptools that the root package is in the python directory
    package_dir={"": "."},
    # Bundle the native gRPC server binary built into pipeann/_bin/.
    package_data={"pipeann": ["_bin/*"]},
    include_package_data=True,
    ext_modules=[CMakeExtension("pipeann.C", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    entry_points={
        "console_scripts": [
            "pipeann-server=pipeann.server:main",
        ],
    },
    zip_safe=False,
    python_requires=">=3.7",
)
