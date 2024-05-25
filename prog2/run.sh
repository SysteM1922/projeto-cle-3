nvcc -O2 -Wno-deprecated-gpu-targets -o a main.cu -lm
./a $@