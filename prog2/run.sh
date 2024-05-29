nvcc -O3 -Wno-deprecated-gpu-targets -o a main.cu -lm
./a $@