# ---------------------------- #
# -- SortieDeCache Makefile -- #
# ---------------------------- #

# -- File list ----------
FILE = openmp.cpp


# -- Paths ----------
INC_PATH = include

# -- Macros ----------
CC = g++

# -- Flags ----------
# No Vectorize
C_OPTIMISATION_FLAGS = -O3 -std=c++17 -march=native -fopenmp -fno-tree-vectorize

C_INC_FLAGS = -I$(INC_PATH)

CFLAGS = $(C_OPTIMISATION_FLAGS) $(C_INC_FLAGS)
LDFLAGS = $(C_OPTIMISATION_FLAGS) $(C_INC_FLAGS)

# -- Final product ----------
PRODUCT = openmp.run

# -- src and obj List ----------
SRC = $(FILE)
OBJ = $(addsuffix .o, $(basename $(FILE)))

# -- Base rules ----------
%.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

#-----Main rule ----------
$(PRODUCT): $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS) $(INC) -lm

# -- Cleanup ----------
clean:
	rm -f $(OBJ)
	rm -f $(PRODUCT)