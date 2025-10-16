# 记录

## GGML环境环境

* Python环境

使用 conda 虚拟环境
```BASH
$ conda active py312
$ cd ggml
$ pip install -r requirements.txt
$ conda install cmake
```

* GGML_NATIVE
  * 如果 GGML_NATIVE 被定义且为真，并且 CUDA Toolkit 版本 ≥ 11.6 且 CMake 版本 ≥ 3.24：
CMake 会设置 CMAKE_CUDA_ARCHITECTURES 为 "native"，即只编译本机可用的 GPU 架构（自动检测当前机器上的 GPU）。这样生成的 CUDA 二进制只适用于当前编译机器上的 GPU，编译速度快，生成文件体积小，但可移植性较差。
  * 如果没有定义 GGML_NATIVE：CMake 会为多个主流架构生成 PTX 或二进制（如 50-virtual, 70-virtual, 86-real 等），这样编译出来的库可以在更多不同型号的 GPU 上运行，移植性更好，但编译时间和文件体积会增加。

* RTX5090 带卡编译
  * 需要安装cmake 3.24以上的版本，所以选择conda环境，放弃ggml自带的venv方式。
  * GGML_NATIVE=ON

```BASH
$ mkdir build && cd build
# fix the path to point to your CUDA compiler
$ cmake -DGGML_CUDA=ON -DGGML_NATIVE=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
$ cmake --build . --config Release -j 24
```

* 无卡编译
  * GGML_NATIVE=OFF
  * 需要扩展一下cmakelist文件，set(CMAKE_CUDA_ARCHITECTURES "120")
  * 参考：[CMAKE_CUDA_ARCHITECTURES](!https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html#variable:CMAKE_CUDA_ARCHITECTURES)

```BASH
$ mkdir build && cd build
# fix the path to point to your CUDA compiler
$ cmake -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
$ cmake --build . --config Release -j 24
```

## CUDA编译

我在RTX5090环境上指定`GGML_NATIVE`进行GGML编译，查看编译后的文件：

```bash
$ cuobjdump --list-elf ./softmax.cu.o

>> ELF file    1: softmax.cu.1.sm_120.cubin
```
那么可以判断这种方式下将直接生成machine code 而不会生成PTX code，并且也确定了cmake正确检测到了当前的 RTX5090对应的`arch = sm_120`.
cuobjdump还有其他的一些有用选项查询ptx code 或者 asm code：
```bash
$ cuobjdump --dump-ptx ./softmax.cu.o
$ cuobjdump --dump-sass ./softmax.cu.o
```

## Add AC4K Backend 
