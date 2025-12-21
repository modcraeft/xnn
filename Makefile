CXX      := g++
CC       := gcc

CXXFLAGS := -O3 -Wall -Wextra -fpermissive -Ilibs -Ilibs/imgui -Ilibs/implot -Ilibs/imgui/backends -I.
CFLAGS   := -O3 -Wall -Wextra -Ilibs -I.
LDFLAGS  := -lglfw -lGL -ldl -lX11 -lm
SDLFLAGS := $(shell pkg-config --cflags --libs sdl2 2>/dev/null || echo -lSDL2)

BUILD    := build
TARGET   := $(BUILD)/xnn
IMGUI_LIB := $(BUILD)/libimgui.so

# ImGui sources
IMGUI_SRC := libs/imgui/imgui.cpp \
             libs/imgui/imgui_draw.cpp \
             libs/imgui/imgui_tables.cpp \
             libs/imgui/imgui_widgets.cpp \
             libs/imgui/imgui_demo.cpp \
             libs/imgui/backends/imgui_impl_glfw.cpp \
             libs/imgui/backends/imgui_impl_opengl3.cpp \
             libs/implot/implot.cpp \
             libs/implot/implot_items.cpp

$(BUILD):
	@mkdir -p $@

$(IMGUI_LIB): $(IMGUI_SRC) | $(BUILD)
	@echo "Building libimgui.so"
	$(CXX) -shared -fPIC $(CXXFLAGS) $^ -o $@ -lglfw -lGL

# Main
$(TARGET): src/main.cpp libs/glad/glad.c $(IMGUI_LIB) plugin.h | $(BUILD)
	@echo "Building xnn"
	$(CXX) $(CXXFLAGS) src/main.cpp libs/glad/glad.c -o $@ \
		-L$(BUILD) -limgui \
		$(LDFLAGS) -rdynamic \
		-Wl,-rpath,'$$ORIGIN'

# Plugins — rpath to find libimgui.so
PLUGIN_SRCS    := $(wildcard plugins/*.cpp)
PLUGIN_TARGETS := $(patsubst plugins/%.cpp, plugins/%.so, $(PLUGIN_SRCS))

plugins/%.so: plugins/%.cpp plugin.h $(IMGUI_LIB)
	@echo "Building plugin: $@"
	$(CXX) -shared -fPIC $(CXXFLAGS) \
		-Wl,-rpath,'$$ORIGIN/../build' \
		$< -L$(BUILD) -limgui -o $@

plugins: $(PLUGIN_TARGETS)

# Demos
$(BUILD)/demos/%: demos/%.c | $(BUILD)/demos
	$(CC) $(CFLAGS) $< -o $@ $(SDLFLAGS) -lm

$(BUILD)/demos:
	mkdir -p $@

demos: $(BUILD)/demos $(wildcard demos/*.c)
	$(MAKE) $(patsubst demos/%.c,$(BUILD)/demos/%,$(wildcard demos/*.c))

all: $(TARGET) plugins demos
	@echo "xnn build complete!"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD) plugins/*.so

reload: clean all run

.PHONY: all run clean demos plugins reload
