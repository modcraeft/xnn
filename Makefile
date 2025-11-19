# ==============================================================
# xnn Makefile – header-only library + multi-demo build
# ==============================================================
CC := gcc
CFLAGS := -Wall -Wextra -O2 -I. -I/usr/include/SDL2 -Ilibs
LIBS := -lm -lSDL2
# Directories
SRC_DIR := demos
BUILD_DIR:= build
# Source files
SOURCES := $(wildcard $(SRC_DIR)/*.c)
# Object files
OBJECTS := $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
# Executables
BINS := $(SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%)
# Default target
all: $(BUILD_DIR) $(BINS)
# Create build directory
$(BUILD_DIR):
	@mkdir -p $@
# Compile each demo → .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c xnn.h
	$(CC) $(CFLAGS) -c $< -o $@
# Link each demo → executable
$(BUILD_DIR)/%: $(BUILD_DIR)/%.o xnn.h
	$(CC) $(CFLAGS) $< -o $@ $(LIBS)
# Clean
clean:
	rm -rf $(BUILD_DIR)
# Run specific demo
run-%: $(BUILD_DIR)/%
	@echo "=== Running $* ==="
	@$<
.PHONY: all clean
