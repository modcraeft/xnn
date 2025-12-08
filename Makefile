CXX      := g++
CC       := gcc

# Main app flags
CXXFLAGS := -O3 -Wall -Wextra -fpermissive -Ilibs -Ilibs/imgui -Ilibs/imgui/backends -I.
CFLAGS   := -O3 -Wall -Wextra -Ilibs -I.

LDFLAGS  := -lglfw -lGL -ldl -lX11 -lm
SDLFLAGS := $(shell pkg-config --cflags --libs sdl2 2>/dev/null || echo -lSDL2)

BUILD    := build
TARGET   := $(BUILD)/xnn
DEMO_DIR := $(BUILD)/demos

# ImGui sources (used by both main app and plugins)
IMGUI_SRC := libs/imgui/imgui.cpp \
             libs/imgui/imgui_draw.cpp \
             libs/imgui/imgui_tables.cpp \
             libs/imgui/imgui_widgets.cpp \
             libs/imgui/imgui_demo.cpp \
             libs/imgui/backends/imgui_impl_glfw.cpp \
             libs/imgui/backends/imgui_impl_opengl3.cpp

# Main application
$(TARGET): src/main.cpp libs/glad/glad.c $(IMGUI_SRC) plugin.h | $(BUILD)
	$(CXX) $(CXXFLAGS) src/main.cpp libs/glad/glad.c $(IMGUI_SRC) -o $@ $(LDFLAGS)

# Hot-reloadable plugins (*.so)
PLUGIN_SRCS   := $(wildcard plugins/*.cpp)
PLUGIN_TARGETS:= $(patsubst plugins/%.cpp, plugins/%.so, $(PLUGIN_SRCS))

# Rule to build each plugin — links full ImGui inside
plugins/%.so: plugins/%.cpp plugin.h $(IMGUI_SRC)
	@echo "Building plugin: $@"
	$(CXX) -shared -fPIC $(CXXFLAGS) $< $(IMGUI_SRC) -o $@

plugins: $(PLUGIN_TARGETS)


# C demos
DEMO_SRCS    := $(wildcard demos/*.c)
DEMO_TARGETS := $(patsubst demos/%.c, $(DEMO_DIR)/%, $(DEMO_SRCS))

$(DEMO_DIR)/%: demos/%.c | $(DEMO_DIR)
	@echo "Building demo: $@"
	@if grep -qE "#[[:space:]]*include[[:space:]]*[<'\"](SDL2?/|SDL\.h)" $<; then \
		$(CC) $(CFLAGS) $< -o $@ $(SDLFLAGS) -lm; \
	else \
		$(CC) $(CFLAGS) $< -o $@ -lm; \
	fi

demos: $(DEMO_TARGETS)


# Utility targets
all: $(TARGET) plugins demos

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD) plugins/*.so

$(BUILD) $(DEMO_DIR):
	mkdir -p $@

# Helpful aliases
reload:
	@echo "Reload"

.PHONY: all run clean demos plugins reload
