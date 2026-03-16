# iCUDA

iCUDA is an attempt at recreating CUDA but for all integrated graphic gpus

### Why
most python scripts use your computers cpu if it doesnt detect if you have a actual gpu or a nividia gpu that supports CUDA

### How It Works
iCUDA works through Extreme Vectorization Normally, a CPU processes one piece of data at a time. iCUDA uses AVX2 (Advanced Vector Extensions 2) to force the silicon to process 8 floating-point numbers in a single clock cycle.

1.  **SIMD (Single Instruction, Multiple Data):** We use `__m256` registers to pack 8 different calculations into one instruction.
2.  **FMA (Fused Multiply-Add):** Instead of multiplying then adding (two steps), iCUDA uses `_mm256_fmadd_ps`. This is exactly what high-end NVIDIA GPUs do—it combines the math into one hardware operation, effectively doubling the math throughput.
3.  **Sparse Memory Access:** The C core skips near-zero activations. Instead of wasting time calculating "0 * Weight," it detects empty signals in the model and jumps to the next calculation, saving millions of cycles per second.

---

### Implementation
To use iCUDA in your project, you use the `ctypes` library to map Python data directly into the C-Vector core.

```python
import ctypes
import numpy as np

# Bind the iCUDA DLL
dll = ctypes.CDLL("./icuda.dll")

# Define the C-Signature for the Vector Optimizer
dll.update_output_layer.argtypes = [
    ctypes.POINTER(ctypes.c_float), # Weight Memory
    ctypes.POINTER(ctypes.c_float), # Hidden State
    ctypes.POINTER(ctypes.c_float), # Grad Signal
    ctypes.c_int,                   # Hidden Dim
    ctypes.c_int,                   # Vocab Size
    ctypes.c_float                  # Learning Rate
]

# Helper to pass Numpy arrays to C-Silicon
def ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Run the accelerated update
dll.update_output_layer(ptr(W2), ptr(h), ptr(grads), 256, 154856, 0.001)
```

---

### Training Benchmarks
**Hardware:** Intel(R) Core(TM) i3-1315U (13th Gen) | 6 Cores | 8 Logical Processors
**Model Size:** ~154,856 Vocabulary LLM(lite language model not large language model)

| Version | Speed (TPS) | Total Dataset ETA | Performance Gain |
| :--- | :--- | :--- | :--- |
| **Initial (Standard CPU)** | ~1.5 TPS | 05:51:10 | Baseline |
| **iCUDA** | **40.1 TPS** | **01:43:43** | **~2,600% Speedup** |

---

### Source & Compilation
iCUDA is completely open-source and safe. It requires no installers or third-party libraries. You compile it yourself so you know exactly what code is running on your hardware.

#### 1. Setup the Compiler
1. Go to [WinLibs.com](https://winlibs.com/#download-release).
2. Scroll to the "Latest" section.
3. Find **"Win64 (without LLVM/Clang/LLD/LLDB): Zip Archive"** under the **UCRT** release. 
   * *Direct Link:* [GCC 15.2.0 + MinGW-w64 13.0.0](https://github.com/brechtsanders/winlibs_mingw/releases/download/15.2.0posix-13.0.0-ucrt-r6/winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64ucrt-13.0.0-r6.zip)
4. Extract the folder. Go into the `bin` folder and run `gcc.exe`(or whatever the fuck the main file looks like in the folder) should take less than half a second for it to setup then to make sure its installed open a cmd and type gcc -v and if you see a bunch of random bullshit and not an error its downloaded

#### 2. Compile iCUDA
Once you have the `icuda.c` file and GCC ready, run

```bash
gcc -shared -o icuda.dll icuda.c -O3 -mavx2 -mfma -static
```
