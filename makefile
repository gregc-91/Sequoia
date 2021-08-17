
# Config options
EXE    = raytracer
SRC    = src
BIN    = bin
CC     = g++
SHELL  = /bin/bash
CFLAGS = -std=c++17 -march=native -fopenmp -O3 -Wall -g3 -gdwarf-2 -Wno-ignored-attributes
ifeq ($(OS), Windows_NT)
	CFLAGS += -static
endif

# Shell colours
BLACK =\u001b[30;1m
RED   =\u001b[31;1m
GREEN =\u001b[32;1m
YELLOW=\u001b[33;1m
BLUE  =\u001b[34;1m
PINK  =\u001b[35;1m
CYAN  =\u001b[36;1m
WHITE =\u001b[37;1m
RESET =\u001b[0m

# List all the header files
HDRS = $(shell find $(SRC) -print | grep .h)

# List all the object files
OBJS = $(shell find $(SRC) -print | grep .cpp | sed -r "s/($(SRC))\/(.*)\.(cpp)/$(BIN)\/obj\/\2\.o/")

# Rules
.PHONY: clean
.SILENT: $(BIN)/$(EXE) $(OBJS) $(BIN) clean

# Rule to create the executable
$(BIN)/$(EXE): $(OBJS)
	$(CC) -o $@ $^ $(CFLAGS) && \
	echo -e "Make: successfully built executable ${CYAN}$@${RESET}"

# Rule to create the object files
$(BIN)/obj/%.o: $(SRC)/%.cpp $(HDRS) | $(BIN)
	echo -e "Make: compiling source file ${PINK}$<${RESET}" && \
	$(CC) -c -o $@ $< $(CFLAGS)

# Rule to create the output folder
$(BIN):
	mkdir -p $(BIN)/obj

# Delete the objects and executable
clean:
	echo -e "Make: cleaning up" && \
	rm -rf $(BIN)
