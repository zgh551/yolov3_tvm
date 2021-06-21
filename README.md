# mnist_tvm
This project using **TVM** to deploy the mnist module on pc or arm device.

## 1. Build
- `x64`

1. in `mnist_tvm` folder create `build_x64` and enter into it

```shell
$ mkdir build_x64
$ cd build_x64
```

2. configure build environment

set the `CMAKE_SYSTEM_PROCESSOR=x64`to link the `x64` runtime dynamic library.

```shell
$ cmake .. -DCMAKE_SYSTEM_PROCESSOR=x64
```

3. build the source file

this step will generate `mnist_test` executable file.

```shell
$ make -j$(nproc)
```


- `armv8`

1. install cross-compile for `aarch64`

```shell
$ sudo apt-get update
$ sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

2. in `mnist_tvm` folder create `build_armv8` and enter into it.

```shell
$ mkdir build_armv8
$ cd build_armv8
```

3. configure build environment

```shell
$ cmake .. \
		-DCMAKE_SYSTEM_NAME=Linux \
		-DCMAKE_SYSTEM_VERSION=1 \
		-DCMAKE_SYSTEM_PROCESSOR=armv8 \
		-DCMAKE_C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc \
		-DCMAKE_CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++ \
		-DCMAKE_FIND_ROOT_PATH=/usr/aarch64-linux-gnu \
		-DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
		-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
		-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH \
```
3. build the project
```shell
$ make -j$(nproc)
```

## 2. Running

when running the executable file,three parameter need to be input.

```shell
executer [image path] [module dynamic lib path] [module parameter path]
```

## 3. Memory Space

when using `opencl` deploy the module, `cpu`as host send the data to device. so we need using `TVMArrayCopyFromBytes`and `TVMArrayCopyToBytes`to realize memory copy.




## Dependence

### OpenCV 

- Compilation Tool

1. GCC 
2. CMAKE

- 3rdparty_lib

1. ffmpeg
2. GTK + 2.x
3. libav : libavcodec-dev libavformat-dev libswscale-dev

- optional package

1. libtbb2 libtbb-dev
2. libjpeg-dev, libpng-dev, libtiff-dev, libjasper-dev, libdc1394-22-dev
3. 

1. install third-party-lib 

```
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libatlas-base-dev gfortran libgtk2.0-dev
```



```shell
$ cmake -D CMAKE_CXX_COMPILER=/usr/bin/arm-linux-gnueabihf-g++ \
		-D CMAKE_C_COMPILER=/usr/bin/arm-linux-gnueabihf-gcc \
		-D CMAKE_BUILD_TYPE=RELEASE \
    	-D CMAKE_INSTALL_PREFIX=/usr/local \
    	-D INSTALL_C_EXAMPLES=ON \
    	-D INSTALL_PYTHON_EXAMPLES=ON \
    	-D OPENCV_GENERATE_PKGCONFIG=ON \
    	-D OPENCV_EXTRA_MODULES_PATH=~/opencv_build/opencv_contrib/modules \
    	-D BUILD_EXAMPLES=ON ..
```

#### Cross Compiler

- GCC install

1. ARM32

```shell
$ sudo apt-get install g++-arm-linux-gnueabihf gcc-arm-linux-gnueabihf
```

2. ARM64

```shell
$ sudo apt-get install g++-aarch64-linux-gnu gcc-aarch64-linux-gnu
```



`hf`: Hard float 

base on the actual processor select whether surport `hf`.



- Using default `cmake`

in the folder `opencv/platforms/linux`, find **aarch64-gnu.toolchain.cmake** and **arm-gnueabi.toolchain.cmake**,

1. ARMv7

```shell
$ cmake -D CMAKE_TOOLCHAIN_FILE="/path/to/opencv/platforms/linux/arm-gnueabi.toolchain.cmake" ..
```

1. ARMv8 

```shell
$ cmake -D CMAKE_TOOLCHAIN_FILE="/path/to/opencv/platforms/linux/aarch64-gnu.toolchain.cmake" ..
```

- System Name

|       |         |      |           |              |           |      |
| ----- | ------- | ---- | --------- | ------------ | --------- | ---- |
| Linux | Android | QNX  | WindowsCE | WindowsPhone | Windows10 |      |

- System Processor

1. arm
2. x86

- compiler error

1. Disable `BUILD_opencv_freetype=OFF`can solve follow error.

````
/usr/lib/gcc-cross/aarch64-linux-gnu/7/../../../../aarch64-linux-gnu/bin/ld: cannot find -lfreetype
/usr/lib/gcc-cross/aarch64-linux-gnu/7/../../../../aarch64-linux-gnu/bin/ld: cannot find -lharfbuzz
````

2. add `freetype `lib using cross compiler tool



