#define XNN_IMPLEMENTATION
#include "../xnn.h"
#include "patterns.h"
#include <SDL2/SDL.h>
#include <stdio.h>
#include <math.h>

#define TILE     20
#define PAD      50
#define WIDTH    (2 * (TRAIN_DIM * TILE + PAD) + PAD)
#define HEIGHT   (2 * (TRAIN_DIM * TILE + PAD) + 130)

#define INPUT_NEURONS  1
#define HIDDEN         128
#define OUTPUT_NEURONS (TRAIN_DIM * TRAIN_DIM)

size_t arch[] = { INPUT_NEURONS, HIDDEN, HIDDEN, OUTPUT_NEURONS };
int    acts[] = { 0, ACT_RELU, ACT_RELU, ACT_SIGMOID };
#define LAYERS ARRAY_LEN(arch)

static Network *net  = NULL;
static Network *grad = NULL;

static float input_val     = 1.0f;
static int   mouse_down    = 0;
static int   auto_animate  = 1;
static float anim_t        = 0.0f;
static int   epoch         = 0;
static float loss          = 0.0f;

static Matrix *train_X = NULL;  // 3×1
static Matrix *train_Y = NULL;  // 3×196
static Data   batch;

void prepare_data(void)
{
    train_X = matrix_alloc(3, 1);
    train_Y = matrix_alloc(3, OUTPUT_NEURONS);

    float inputs[3] = {0.0f, 1.0f, 2.0f};

    for (int i = 0; i < 3; i++) {
        train_X->data[i] = inputs[i];

        float (*src)[TRAIN_DIM] = (i == 0) ? train_data :
                                  (i == 1) ? train_data2 : train_data3;

        for (int y = 0; y < TRAIN_DIM; y++) {
            for (int x = 0; x < TRAIN_DIM; x++) {
                float val = src[y][x] ? 0.99f : 0.01f;
                train_Y->data[i * OUTPUT_NEURONS + y*TRAIN_DIM + x] = val;
            }
        }
    }
    batch.in  = train_X;
    batch.out = train_Y;
}

void render_glyph(SDL_Renderer *ren, const float *pixels, int ox, int oy)
{
    for (int y = 0; y < TRAIN_DIM; y++) {
        for (int x = 0; x < TRAIN_DIM; x++) {
            float v = pixels[y*TRAIN_DIM + x];
            int b = (int)(v * 255.0f);
            SDL_SetRenderDrawColor(ren, b, b, 255, 255);
            SDL_Rect r = { ox + x*TILE + 2, oy + y*TILE + 2, TILE-4, TILE-4 };
            SDL_RenderFillRect(ren, &r);
        }
    }
}

void draw(SDL_Renderer *ren)
{
    SDL_SetRenderDrawColor(ren, 12, 14, 28, 255);
    SDL_RenderClear(ren);

    float out0[OUTPUT_NEURONS];
    float out1[OUTPUT_NEURONS];
    float out2[OUTPUT_NEURONS];
    float live[OUTPUT_NEURONS];

    network_predict(net, &train_X->data[0], out0);
    network_predict(net, &train_X->data[1], out1);
    network_predict(net, &train_X->data[2], out2);
    network_predict(net, &input_val, live);

    render_glyph(ren, out0, PAD,                  PAD);
    render_glyph(ren, out1, PAD + TRAIN_DIM*TILE + PAD, PAD);
    render_glyph(ren, out2, PAD,                  PAD + TRAIN_DIM*TILE + PAD);
    render_glyph(ren, live, PAD + TRAIN_DIM*TILE + PAD, PAD + TRAIN_DIM*TILE + PAD);

    // Slider
    int sy = HEIGHT - 100;
    SDL_SetRenderDrawColor(ren, 70, 70, 100, 255);
    SDL_Rect track = { PAD, sy, WIDTH - 2*PAD, 26 };
    SDL_RenderFillRect(ren, &track);

    int kx = PAD + (int)((input_val / 2.0f) * (WIDTH - 2*PAD - 50));
    SDL_SetRenderDrawColor(ren, 0, 240, 255, 255);
    SDL_Rect knob = { kx, sy - 18, 50, 62 };
    SDL_RenderFillRect(ren, &knob);

    printf("\rEpoch %7d  │  Loss %.7f  │  Input %.3f  │  SPACE = pause    ",
           epoch, loss, input_val);
    fflush(stdout);
}

int main(void)
{
    XNN_INIT();
    prepare_data();

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window   *win = SDL_CreateWindow("xnn – Glyph Interpolator",
                                         SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                         WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

    net  = network_alloc(arch, LAYERS, acts, LOSS_MSE);
    grad = network_alloc(arch, LAYERS, acts, LOSS_MSE);
    if (!net || !grad) return 1;

    int running = 1;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = 0;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_SPACE)
                auto_animate = !auto_animate;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)
				running = 0;
            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT)
                mouse_down = 1;
            if (e.type == SDL_MOUSEBUTTONUP)
                mouse_down = 0;
            if (e.type == SDL_MOUSEMOTION && mouse_down) {
                int mx = e.motion.x;
                if (mx >= PAD && mx <= WIDTH - PAD) {
                    input_val = 2.0f * (mx - PAD) / (float)(WIDTH - 2*PAD);
                    if (input_val < 0.0f) input_val = 0.0f;
                    if (input_val > 2.0f) input_val = 2.0f;
                    auto_animate = 0;
                }
            }
        }

        // Train hard and fast
        for (int i = 0; i < 100; i++) {
            backprop(net, grad, &batch);
            apply_grad(net, grad, 0.18f);
        }
        epoch += 100;

        // Accurate loss
        loss = network_mse(net, &batch);

        if (auto_animate) {
            anim_t += 0.012f;
            input_val = 1.0f + sinf(anim_t);
        }

        draw(ren);
        SDL_RenderPresent(ren);
        SDL_Delay(16);
    }

    network_free(net);
    network_free(grad);
    matrix_free(train_X);
    matrix_free(train_Y);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    printf("\n");
    return 0;
}
