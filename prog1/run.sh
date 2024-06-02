nvcc -O3 -Wno-deprecated-gpu-targets -o a $1 -lm
shift 1
./a $@