bin := tone_map

cxx := g++
cxx_flags := -Wall -O3 -D_GNU_SOURCE -std=gnu++11

# NOTE: On my laptop, set to compute_20 and sm_20
gpu_arch := compute_50
gpu_code := sm_50
nvcc := nvcc
nvcc_flags := $(strip \
	--gpu-architecture=$(gpu_arch) \
	--gpu-code=$(gpu_code) \
	-ccbin $(cxx) -O3 -D_GNU_SOURCE \
	-DGPU_ARCH=\"$(gpu_arch)\" \
	-DGPU_CODE=\"$(gpu_code)\" \
)

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
