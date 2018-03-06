bin := tone_map

cxx := g++
cxx_flags := -Wall -g -D_GNU_SOURCE

nvcc := nvcc
nvcc_flags := -gencode=arch=compute_50,code=sm_50 -ccbin $(cxx) -g -D_GNU_SOURCE

inc  := -I./include -I/usr/local/cuda/include
libs := -lcuda -lcudart

cppfiles   := $(wildcard src/*.cpp)
cufiles    := $(wildcard src/*.cu)
cpp_ofiles := $(patsubst src/%.cpp,obj/%.o,$(cppfiles))
cu_ofiles  := $(patsubst src/%.cu,obj/%.cu.o,$(cufiles))

.PHONY: all clean re mrproper

all: $(bin)
	

$(bin): $(cu_ofiles) $(cpp_ofiles)
	$(nvcc) $(nvcc_flags) -o $(bin) $^ $(libs)

obj/%.cu.o: src/%.cu
	$(nvcc) $(nvcc_flags) $(inc) -c $< -o $@

obj/%.o: src/%.cpp
	$(cxx) $(cxx_flags) $(inc) -c $< -o $@

clean:
	rm -f $(bin) obj/*.o

re: clean all

mroprer: re
