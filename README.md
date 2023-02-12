# MiePy_CUDA

**MiePy_CUDA** is a Python module to calculate Mie scattering coefficients (backscatter and extinction coefficient) of spheres by CUDA. The implementation is extremely fast, about **0.045ms/case** in large-scale scenarios. The implementation is modified by "Absorption and Scattering of Light by Small Particles (Bohren & Huffman)".

## Preparation
- CUDA >= 10.2
- PyTorch >= 1.0
- Python >= 3.6

## Installation
- Install suitable **CUDA Toolkit** for compiling **.cu** files, and make sure **CUDA_HOME** in your system environment variable.
- Install PyTorch with CUDA support, because the implementation takes PyTorch as frontend.
- Install **setuptools** package in Python enviroment for execute the **setup.py** file.
- Execute `python setup.py install` to compile and install the package.
- Run `python demo.py` to verify the installation. The correct output is
```shell
tensor([0.0267, 0.0708, 0.2003], dtype=torch.float64) tensor([2.2240, 2.1679, 2.1110], dtype=torch.float64)
tensor([0.0042, 0.0042, 0.0042], dtype=torch.float64) tensor([2.0669, 2.0873, 2.1351], dtype=torch.float64)
tensor([0.0042, 0.0042, 0.0042], dtype=torch.float64) tensor([2.0215, 2.0281, 2.0445], dtype=torch.float64)
```

## 安装
- 安装合适的CUDA Toolkit用于编译.cu文件，并且确保CUDA_HOME存在于你的系统环境变量。
- 安装CUDA支持的PyTorch，因为我们采用它作为数据设备间交互的前端。
- 安装setuptools包到Python环境，它用于执行setup.py文件。
- 执行`python setup.py install`来编译和安装。
- 执行`python demo.py`来验证安装是否成功，正确的输出如下面所示：
```shell
tensor([0.0267, 0.0708, 0.2003], dtype=torch.float64) tensor([2.2240, 2.1679, 2.1110], dtype=torch.float64)
tensor([0.0042, 0.0042, 0.0042], dtype=torch.float64) tensor([2.0669, 2.0873, 2.1351], dtype=torch.float64)
tensor([0.0042, 0.0042, 0.0042], dtype=torch.float64) tensor([2.0215, 2.0281, 2.0445], dtype=torch.float64)
```

