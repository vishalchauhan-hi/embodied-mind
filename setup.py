from setuptools import setup, find_packages

setup(
    name="embodied-mind",
    version="0.2.0",
    description="Minecraft agent with VLM perception, LLM planning and episodic memory on MineDojo",
    author="Vishal Chauhan",
    author_email="vishalutokyo@gmail.com",
    url="https://github.com/vishalchauhan-hi/embodied-mind",
    license="MIT",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "Pillow>=9.0",
        "pyyaml>=6.0",
        "requests>=2.28",
    ],
    extras_require={
        # MineDojo 0.1 uses np.unicode_, which NumPy 2 removed
        "minedojo": ["minedojo", "numpy<2"],
        "gemini": ["google-genai"],
        "openai": ["openai>=1.0"],
        "dev": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "embodied-mind=embodied_mind.run:main",
        ],
    },
)
