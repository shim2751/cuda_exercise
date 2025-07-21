NVCC = nvcc
TARGET = bin/run
SOURCES = conv.cu main.cpp

$(TARGET): $(SOURCES)
	mkdir -p bin
	$(NVCC) -O0 -arch=sm_75 $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: clean