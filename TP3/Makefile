# -------------------- #
# -- myRem Makefile -- #
# -------------------- #
FILE= utils.cu
CUDAFILE= main.cu 1-dotp/dotp.cu 2-sobel/sobel.cu 3-transpo/transpo.cu 4-histo/histo.cu

CUDAFILE+= ios.cu pixmap_io.cu container.cu

SRC_PATH = src
OBJ_PATH = obj
MODULE_PATH = exe
INC_PATH = include

CC = g++-6
NVCC = nvcc -ccbin $(CC)

CXXFLAGS = -std=c++11 -Xcompiler -fopenmp
CUDAFLAGS = arch=compute_50,code=sm_50
# CUDAFLAGS = arch=compute_30,code=sm_30
# -Xptxas -dscm=cg
CXX_INC_FLAGS = -I$(INC_PATH)
OPTFLAGS = -O3
TARGET = tpcuda.run

# -- src and obj List ----------
OBJ = $(addprefix ${OBJ_PATH}/, $(addsuffix .o, $(basename $(FILE))))
OBJ += $(addprefix ${OBJ_PATH}/, $(addsuffix .o, $(basename $(CUDAFILE))))

# -- Base rules ----------
$(OBJ_PATH)/%.o : $(SRC_PATH)/%.cu
	$(NVCC) -I /usr/local/cuda/include $(CXX_INC_FLAGS) -dc $< -o $@ 
	 # --ptxas-options=-v


#-----Main rule ----------
$(MODULE_PATH)/$(TARGET): $(OBJ)
	$(NVCC) -gencode $(CUDAFLAGS) -I /usr/local/cuda/include -o $@ $^ $(CXXFLAGS) $(OPTFLAGS) $(CXX_INC_FLAGS) $(LD_FLAGS) 

clean:
	rm -f $(OBJ)
	rm -f exe/*

debug:
	/usr/local/cuda/bin/nvvp $(MODULE_PATH)/$(TARGET)