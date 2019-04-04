# ---------------------------- #
# -- SortieDeCache Makefile -- #
# ---------------------------- #

# -- File list ----------
FILE = SIMD.cpp


# -- Paths ----------
INC_PATH = include

# -- Macros ----------
CC = g++

# -- Flags ----------
# No Vecto
# C_OPTIMISATION_FLAGS = -O3 -std=c++17 -march=native -fno-tree-vectorize

# Vecto invisible
C_OPTIMISAT	ION_FLAGS = -O3 -std=c++17 -march=native

# Forced vecto
# C_OPTIMISATION_FLAGS = -O3 -std=c++17 -march=native -ftree-vectorize -fopt-info

C_INC_FLAGS = -I$(INC_PATH)

CFLAGS = $(C_OPTIMISATION_FLAGS) $(C_INC_FLAGS)
LDFLAGS = $(C_OPTIMISATION_FLAGS) $(C_INC_FLAGS)

# -- Final product ----------
PRODUCT = simd.run

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