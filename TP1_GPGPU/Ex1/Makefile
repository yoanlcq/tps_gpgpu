################################################################################
#
# Build script for project
#
################################################################################
# $(BIN) est la nom du binaire généré
BIN=deviceProperties
# FLAG
FLAGS=-Wall -O3 -std=c++11
FLAGS_NVCC=
# INCLUDES
INC=-I/usr/local/cuda/include
# SRC Directory
SRC=src
# Librairies directory
LIBDIR=
# Librairies
LIBS=-lcuda -lcudart
# Librairies
OBJDIR=obj
# OBJET Directory
NVCC=nvcc
GPP=g++
# $(OBJECTS) sont les objets des fichiers C / C++ standard qui seront générés après la compilation
OBJECTS=$(OBJDIR)/main.o
# $(OBJECTS_CUDA) sont les objets des fichiers contenant du CUDA (.cu)
OBJECTS_CUDA=$(OBJDIR)/deviceProperties.cu.o

all: $(OBJECTS) $(OBJECTS_CUDA)
	@echo "**** PHASE DE LIEN ****"
	$(NVCC) $(LIBDIR) $(LIBS) $(OBJECTS) $(OBJECTS_CUDA) -o $(BIN)

$(OBJDIR)/%.o: $(SRC)/%.cpp
	@echo "**** $@ ****"
	$(GPP) $(INC) $(FLAGS) -c $< -o $@

$(OBJDIR)/%.cu.o: $(SRC)/%.cu
	@echo "**** $@ ****"
	$(NVCC) $(FLAGS_NVCC) $(INC) -c $< -o $@

clean:
	- rm $(OBJDIR)/*.o
	- rm $(BIN)


