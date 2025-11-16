#define XNN_IMPLEMENTATION
#include "xnn.h"
#include <stdio.h>
#include <stdbool.h>
#include <SDL.h>
#include "glyphs.h"

#define WX 0
#define WY 0
#define WW 1200
#define WH 600
#define GRID_SIZE 28
#define CELL_SIZE 20
#define DRAW_X 20
#define DRAW_Y 20
#define DRAW_W (GRID_SIZE * CELL_SIZE)
#define DRAW_H (GRID_SIZE * CELL_SIZE)
#define PRED_X (DRAW_X + DRAW_W + 40)
#define PRED_Y 100
#define GLYPH_SCALE 10
#define BAR_W 20
#define BAR_MAX_H 200
#define BAR_GAP 40

int main() {
    XNN_INIT();

    // Load the model
    size_t arch[] = {784, 128, 10};
    int act[] = {ACT_RELU, ACT_RELU, ACT_SOFTMAX};
    Network *net = network_load("mnist_model.bin", arch, 3, act, LOSS_CE);
    if (!net) {
        fprintf(stderr, "Failed to load model from mnist_model.bin\n");
        return 1;
    }

    SDL_Window *w;
    SDL_Renderer *r;
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        network_free(net);
        return 1;
    }
    w = SDL_CreateWindow("MNIST Digit Recognizer", WX, WY, WW, WH, SDL_WINDOW_SHOWN);
    if (!w) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        network_free(net);
        return 1;
    }
    r = SDL_CreateRenderer(w, -1, SDL_RENDERER_ACCELERATED);
    if (!r) {
        fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(w);
        SDL_Quit();
        network_free(net);
        return 1;
    }

    float canvas[GRID_SIZE][GRID_SIZE] = {0};
    float input[784];
    float output[10];
    bool quit = false;
    bool drawing = false;
    SDL_Event e;

    while (!quit) {
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_QUIT: quit = true; break;
                case SDL_MOUSEBUTTONDOWN:
                    if (e.button.button == SDL_BUTTON_LEFT) drawing = true;
                    break;
                case SDL_MOUSEBUTTONUP:
                    if (e.button.button == SDL_BUTTON_LEFT) drawing = false;
                    break;
                case SDL_KEYDOWN:
                    if (e.key.keysym.sym == SDLK_ESCAPE) quit = true;
                    else if (e.key.keysym.sym == SDLK_SPACE || e.key.keysym.sym == SDLK_c) {
                        memset(canvas, 0, sizeof(canvas));
                    }
                    break;
            }
        }

        // Handle drawing
        int mx, my;
        Uint32 buttons = SDL_GetMouseState(&mx, &my);
        if (drawing && (buttons & SDL_BUTTON(SDL_BUTTON_LEFT))) {
            int gx = (mx - DRAW_X) / CELL_SIZE;
            int gy = (my - DRAW_Y) / CELL_SIZE;
            if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
                canvas[gy][gx] = 1.0f;
                // Make brush thicker: set neighbors
                if (gx > 0) canvas[gy][gx-1] = fmaxf(canvas[gy][gx-1], 0.5f);
                if (gx < GRID_SIZE-1) canvas[gy][gx+1] = fmaxf(canvas[gy][gx+1], 0.5f);
                if (gy > 0) canvas[gy-1][gx] = fmaxf(canvas[gy-1][gx], 0.5f);
                if (gy < GRID_SIZE-1) canvas[gy+1][gx] = fmaxf(canvas[gy+1][gx], 0.5f);
            }
        }

        // Prepare input and run inference
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int x = 0; x < GRID_SIZE; ++x) {
                input[y * GRID_SIZE + x] = canvas[y][x];
            }
        }
        network_predict(net, input, output);

        // Find predicted digit
        int pred_digit = 0;
        float max_prob = output[0];
        for (int i = 1; i < 10; ++i) {
            if (output[i] > max_prob) {
                max_prob = output[i];
                pred_digit = i;
            }
        }

        // Render
        SDL_SetRenderDrawColor(r, 0, 0, 0, 255); // Black background
        SDL_RenderClear(r);

        // Draw grid
        SDL_SetRenderDrawColor(r, 50, 50, 50, 255); // Grid lines
        for (int i = 0; i <= GRID_SIZE; ++i) {
            SDL_RenderDrawLine(r, DRAW_X + i * CELL_SIZE, DRAW_Y, DRAW_X + i * CELL_SIZE, DRAW_Y + DRAW_H);
            SDL_RenderDrawLine(r, DRAW_X, DRAW_Y + i * CELL_SIZE, DRAW_X + DRAW_W, DRAW_Y + i * CELL_SIZE);
        }

        // Draw canvas
        for (int y = 0; y < GRID_SIZE; ++y) {
            for (int x = 0; x < GRID_SIZE; ++x) {
                Uint8 intensity = (Uint8)(255 * canvas[y][x]);
                SDL_SetRenderDrawColor(r, intensity, intensity, intensity, 255);
                SDL_Rect rect = {DRAW_X + x * CELL_SIZE, DRAW_Y + y * CELL_SIZE, CELL_SIZE, CELL_SIZE};
                SDL_RenderFillRect(r, &rect);
            }
        }

        // Draw predicted digit using glyphs (large)
        char digit_char = '0' + pred_digit;
        for (int gy = 0; gy < 9; ++gy) {
            for (int gx = 0; gx < 9; ++gx) {
                if (glyphs[(int)digit_char][gy][gx]) {
                    SDL_SetRenderDrawColor(r, 0, 255, 0, 255); // Green
                    SDL_Rect rect = {PRED_X + gx * GLYPH_SCALE, PRED_Y + gy * GLYPH_SCALE, GLYPH_SCALE, GLYPH_SCALE};
                    SDL_RenderFillRect(r, &rect);
                }
            }
        }

        // Draw probability bars below the digit
        int bar_y = PRED_Y + 9 * GLYPH_SCALE + 50;
        for (int i = 0; i < 10; ++i) {
            int bar_h = (int)(output[i] * BAR_MAX_H);
            SDL_SetRenderDrawColor(r, 0, 255, 0, 255);
            SDL_Rect bar = {PRED_X + i * BAR_GAP, bar_y - bar_h, BAR_W, bar_h};
            SDL_RenderFillRect(r, &bar);

            // Draw label using glyphs (small)
            char label_char = '0' + i;
            for (int gy = 0; gy < 9; ++gy) {
                for (int gx = 0; gx < 9; ++gx) {
                    if (glyphs[(int)label_char][gy][gx]) {
                        SDL_SetRenderDrawColor(r, 255, 255, 255, 255); // White
                        SDL_Rect rect = {PRED_X + i * BAR_GAP + gx, bar_y + 10 + gy, 1, 1};
                        SDL_RenderFillRect(r, &rect);
                    }
                }
            }
        }

        SDL_RenderPresent(r);
    }

    SDL_DestroyRenderer(r);
    SDL_DestroyWindow(w);
    SDL_Quit();
    network_free(net);
    return 0;
}
