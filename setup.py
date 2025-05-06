import re
import sys
import os
from setuptools import find_packages, setup

# Define all runtime dependencies here.
# Development/testing dependencies are excluded.
# torch and torchvision are listed but excluded from install_requires,
# assuming they are handled by the environment or parent package.
_deps = [
    'torch',
    'torchvision',
    'einops>=0.6.0',
    'numexpr>=2.8.4',
    'matplotlib>=3.7.1',
    'pandas>=1.5.3',
    'av>=10.0.0',
    'pims>=0.6.1',
    'imageio-ffmpeg>=0.4.8',
    'rich>=13.3.2',
    'gdown>=4.7.1',
    'py3d>=0.0.87',
    'librosa>=0.10.0.post2',
    'numpy>=1.26.4',
    'opencv-python-headless',
    'timm>=0.6.13',
    'transformers>=4.40.2',
    'omegaconf>=2.3.0',
    'aiohttp>=3.9.3',
    'psutil>=5.9.6',
    'clip-interrogator>=0.6.0',
    'streamlit>=1.27.2',
    'torchsde>=0.2.5',
    'fastapi>=0.100.0',
    'diffusers>=0.27.2',
    'accelerate>=0.29.3',
    'python-decouple>=3.8',
    'mutagen>=1.47.0',
    'imageio[ffmpeg]>=2.34.1',
    'xformers>=0.0.26.post1',
    # 'tensorrt>=10.0.1', # Optional/Commented out dependencies
    # 'onnx_graphsurgeon>=0.5.2',
    # 'onnx>=1.16.0',
    # 'zstandard>=0.22.0',
    # 'polygraphy>=0.49.9',
    'kornia>=0.7.2',
    'wheel>=0.43.0', # Generally useful for building/packaging
    'loguru>=0.7.2',
    'scikit-image>=0.21.0',
    'scipy>=1.11.4',
    'segment-anything>=1.0',
    'piexif>=1.1.3',
    'GitPython>=3.1.43',
    'qtpy>=2.4.1',
    'pyqt6>=6.5.0',
    'pyqt6-qt6>=6.5.0',
    'pyqtgraph>=0.13.7',
    'contexttimer>=0.3.3',
    'PyWavelets>=1.1.1',
    'opensimplex>=0.4.2',
    'moviepy==1.0.3',
    'color-matcher>=0.5.0',
    'pydub>=0.23.0'
]

# Helper to parse package names from dependency strings
pattern = re.compile(r"^([^@!=<>~]+)(?:[@!=<>~].*)?$")
deps = {match[0]: x for x in _deps for match in [pattern.findall(x)] if match}

def deps_list(*pkgs):
    """Returns the full dependency string for each package name."""
    return [deps[pkg] for pkg in pkgs if pkg in deps]

# Define the list of package names for core installation requirements
# Excludes 'torch', 'torchvision'
install_requires_packages = [
    'einops', 'numexpr', 'matplotlib', 'pandas', 'av', 'pims', 'imageio-ffmpeg',
    'rich', 'gdown', 'py3d', 'librosa', 'numpy', 'opencv-python-headless',
    'timm', 'transformers', 'omegaconf', 'aiohttp', 'psutil', 'clip-interrogator',
    'streamlit', 'torchsde', 'fastapi', 'diffusers', 'accelerate', 'python-decouple',
    'mutagen', 'imageio[ffmpeg]', 'xformers', 'kornia', 'wheel', 'loguru',
    'scikit-image', 'scipy', 'segment-anything', 'piexif', 'GitPython', 'qtpy',
    'pyqt6', 'pyqt6-qt6', 'pyqtgraph', 'contexttimer', 'PyWavelets',
    'opensimplex', 'moviepy', 'color-matcher', 'pydub'
]
install_requires = deps_list(*install_requires_packages)

setup(
    # --- Package Metadata ---
    name="deforum",
    # Consider updating the version number to reflect these significant changes
    version="0.01.9.dev0",
    description="Core components for Deforum project.", # Updated description slightly
    long_description=open("README.md", "r", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    keywords="deep learning diffusion deforum stable diffusion ai animation",
    license="Apache",
    author="The Deforum team",
    author_email="deforum-art@deforum.com", # Check if this is still the correct contact
    url="https://github.com/Shiba-2-shiba/deforum-studio", # Updated URL to user's repo

    # --- Package Structure ---
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,

    # --- Dependencies ---
    python_requires=">=3.8.0, <3.13",
    install_requires=install_requires,
    # extras_require is removed

    # --- Entry Points ---
    entry_points={
        "console_scripts": [
            # Keep these if they are still relevant for this slimmed-down package
            "deforum=deforum.commands.deforum_cli:start_deforum_cli",
            "deforum-test=deforum.commands.deforum_test:start_deforum_test",
            "deforum-profile=deforum.commands.deforum_profiling:start_deforum_test"
        ]
    },

    # --- Classifiers ---
    classifiers=[
        "Development Status :: 5 - Production/Stable", # Review if appropriate
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", # Added 3.9 explicitly
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11", # Added 3.11 explicitly
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: Editors", # Added potentially relevant topic
        "Topic :: Multimedia :: Video",             # Added potentially relevant topic
    ],
    # cmdclass is removed
)

# Release checklist comments (kept for reference, adapt if needed)
# ... (checklist remains the same) ...
