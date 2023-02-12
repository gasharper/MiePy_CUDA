# MiePy_CUDA
Python module to calculate Mie scattering coefficients by CUDA implementation.

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
