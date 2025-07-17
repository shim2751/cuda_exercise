NVCC = nvcc
TARGET = vec_add

$(TARGET): vec_add.cu main.cpp
	$(NVCC) -O3 -arch=sm_75 vec_add.cu main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean