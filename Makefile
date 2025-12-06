CXX       := g++
CC        := gcc

CXXFLAGS  := -O3 -Wall -Wextra -fpermissive -Ilibs -Ilibs/imgui -Ilibs/imgui/backends
CFLAGS    := -O3 -Wall -Wextra -Ilibs -I.           # finds "../xnn.h" and "xnn.h"
LDFLAGS   := -lglfw -lGL -ldl -lX11 -lm

# Prefer pkg-config for SDL2, fall back to plain -lSDL2
SDLFLAGS  := $(shell pkg-config --cflags --libs sdl2 2>/dev/null || echo -lSDL2)

BUILD     := build
TARGET    := $(BUILD)/xnn
DEMO_DIR  := $(BUILD)/demos

#ImGui sources
IMGUI_SRC := libs/imgui/imgui.cpp \
             libs/imgui/imgui_draw.cpp \
             libs/imgui/imgui_tables.cpp \
             libs/imgui/imgui_widgets.cpp \
             libs/imgui/imgui_demo.cpp \
             libs/imgui/backends/imgui_impl_glfw.cpp \
             libs/imgui/backends/imgui_impl_opengl3.cpp

#Main C++ app
$(TARGET): src/main.cpp libs/glad/glad.c $(IMGUI_SRC) | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

#Automatic demos (in build/demos/)
DEMO_SRCS    := $(wildcard demos/*.c)
DEMO_TARGETS := $(patsubst demos/%.c, $(DEMO_DIR)/%, $(DEMO_SRCS))

demos: $(DEMO_TARGETS)

# Build rule: one command, auto-detects SDL2
$(DEMO_DIR)/%: demos/%.c | $(DEMO_DIR)
	@echo "Building demo: $@"
	@if grep -qE "#[[:space:]]*include[[:space:]]*[<'\"](SDL2?/|SDL\.h)" $<; then \
		$(CC) $(CFLAGS) $< -o $@ $(SDLFLAGS) -lm; \
	else \
		$(CC) $(CFLAGS) $< -o $@ -lm; \
	fi

#Utility targets
run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD) 

$(BUILD) $(DEMO_DIR):
	mkdir -p $@

#Phony targets
.PHONY: all demos run clean
all: $(TARGET) demos
