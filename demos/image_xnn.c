#define XNN_IMPLEMENTATION
#include "xnn.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

// ====================== CONFIG ======================
#define LEARNING_RATE       0.3f
#define BATCH_SIZE          8
#define BATCHES_PER_FRAME   400
#define ARCH                2, 16, 32, 16, 1          // input (x,y) - output brightness
//#define WINDOW_W            1766
//#define WINDOW_H            1405
#define WINDOW_W            1200
#define WINDOW_H            1000
//#define NET_VIEW_W          1110.0f
//#define NET_VIEW_H          820.0f
#define NET_VIEW_W          WINDOW_W * 0.56f
#define NET_VIEW_H          WINDOW_H * 0.56f
//#define COST_VIEW_W         600.0f
//#define COST_VIEW_H         820.0f
#define COST_VIEW_W         WINDOW_W * 0.40f
#define COST_VIEW_H         WINDOW_H * 0.63f
#define VERIFY_SIDE         (WINDOW_H - 100 - NET_VIEW_H)

// Activations: ReLU in hidden layers, Sigmoid on output (0-1)
static const size_t arch[] = { ARCH };
static const int    acts[] = { ACT_RELU, ACT_RELU, ACT_RELU, ACT_SIGMOID };

typedef struct { 
    uint8_t *data; 
    int w, h, c; 
} Image;
Image img = {0};

float  g_cost = 1e9f;
size_t g_epoch = 0;
bool   g_paused = false;
bool   g_fullscreen = false;
float  g_upscale = 1.0f;

#include "glyphs.h"

static void fill_circle(SDL_Renderer *r, int cx, int cy, int rad, Uint8 rr, Uint8 gg, Uint8 bb, Uint8 aa)
{
    SDL_SetRenderDrawColor(r, rr, gg, bb, aa);
    for (int y = -rad; y <= rad; ++y)
        for (int x = -rad; x <= rad; ++x)
            if (x*x + y*y <= rad*rad)
                SDL_RenderDrawPoint(r, cx + x, cy + y);
}

static void render_label(SDL_Renderer *r, int x, int y, float value, int digits, int size)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", digits, value);
    int cell = size;
    SDL_Rect rect = {0};
    rect.w = rect.h = cell - 1;

    SDL_SetRenderDrawColor(r, 0x55, 0x99, 0xFF, 255);
    for (size_t i = 0; i < strlen(buf); ++i) {
        for (int gy = 0; gy < 9; ++gy)
            for (int gx = 0; gx < 9; ++gx)
                if (glyphs[(unsigned char)buf[i]][gy][gx]) {
                    rect.x = x + gx * cell + i * 9 * cell;
                    rect.y = y + gy * cell;
                    fill_circle(r, rect.x + cell/2, rect.y + cell/2, 2, 0x55, 0x99, 0xFF, 255);
                }
    }
}

static void render_network(SDL_Renderer *r, Network *net)
{
    SDL_SetRenderDrawColor(r, 0x33, 0x33, 0x33, 255);
    SDL_Rect border = {20, 20, (int)NET_VIEW_W, (int)NET_VIEW_H};
    SDL_RenderDrawRect(r, &border);

    size_t layers = ARRAY_LEN(arch);
    float layer_w = (NET_VIEW_W - 60) / (layers - 1);

    for (size_t l = 0; l < layers; ++l) {
        float cx = 50 + l * layer_w;
        float neuron_h = NET_VIEW_H / arch[l];
        for (size_t n = 0; n < arch[l]; ++n) {
            float cy = 20 + n * neuron_h + neuron_h/2;

            // connections
            if (l < layers-1) {
                for (size_t t = 0; t < arch[l+1]; ++t) {
                    float tx = 50 + (l+1)*layer_w;
                    float ty = 20 + t * NET_VIEW_H / arch[l+1] + (NET_VIEW_H / arch[l+1])/2;
                    float w = net->w[l]->data[t * arch[l] + n];
                    float val = 0.5f + 0.5f * w * 10.0f; // visual scaling
                    val = val < 0 ? 0 : val > 1 ? 1 : val;
                    Uint8 col = (Uint8)(val * 200 + 55);
                    SDL_SetRenderDrawColor(r, col, col, col, (col/4) + 10);
                    SDL_RenderDrawLine(r, (int)cx, (int)cy, (int)tx, (int)ty);
                }
            }

            // neuron circle
            float bias = (l == 0) ? 0.0f : net->b[l-1]->data[n];
            float act = 0.5f + 0.5f * bias * 5.0f;
            act = act < 0 ? 0 : act > 1 ? 1 : act;
            Uint8 col = (Uint8)(act * 255);
            int rad = (int)(30.0f / arch[l]) + 6;
            if (l == 0 || l == layers-1)
                fill_circle(r, (int)cx, (int)cy, rad > 20 ? 20 : rad, 0x00, 0xFF, 0x00, 255);
            else
                fill_circle(r, (int)cx, (int)cy, rad, col, 0xAA, 0xFF, 255);
        }
    }
}

static void render_cost(SDL_Renderer *r)
{
    int x = 20 + (int)NET_VIEW_W + 20;
    int y = 20;
    SDL_SetRenderDrawColor(r, 0x33, 0x33, 0x33, 255);
    SDL_Rect border = {x, y, (int)COST_VIEW_W, (int)COST_VIEW_H};
    SDL_RenderDrawRect(r, &border);

    // simple bar
    int bar_h = (int)(g_cost * COST_VIEW_H * 0.8f);
    if (bar_h > (int)COST_VIEW_H) bar_h = (int)COST_VIEW_H;
    SDL_SetRenderDrawColor(r, 255, 50, 50, 150);
    SDL_Rect bar = {x + 50, y + COST_VIEW_H - bar_h, (int)COST_VIEW_W - 100, bar_h};
    SDL_RenderFillRect(r, &bar);

    render_label(r, x + 60, y + 400, g_cost, 6, 4);
    render_label(r, x + 60, y + 20, (float)g_epoch, 0, 4);
}

static void render_original(SDL_Renderer *r, int x, int y, int sz)
{
    SDL_Rect border = {x, y, sz, sz};
    SDL_SetRenderDrawColor(r, 0x55, 0x55, 0x55, 255);
    SDL_RenderDrawRect(r, &border);

    int pw = sz / img.w;
    int ph = sz / img.h;
    for (int iy = 0; iy < img.h; ++iy)
        for (int ix = 0; ix < img.w; ++ix) {
            Uint8 v = img.data[iy * img.w + ix];
            SDL_SetRenderDrawColor(r, v, v, v, 255);
            SDL_Rect p = {x + ix*pw + 10, y + iy*ph + 10, pw-1, ph-1};
            SDL_RenderFillRect(r, &p);
        }
}

static void render_prediction(SDL_Renderer *r, int x, int y, int sz, Network *net, bool hi_res)
{
    int scale = hi_res ? (int)g_upscale : 1;
    int w = img.w * scale;
    int h = img.h * scale;
    int pw = sz / w;
    int ph = sz / h;

    SDL_Rect border = {x, y, sz, sz};
    SDL_SetRenderDrawColor(r, 0x55, 0x55, 0x55, 255);
    SDL_RenderDrawRect(r, &border);

    for (int iy = 0; iy < h; ++iy)
        for (int ix = 0; ix < w; ++ix) {
            float nx = (float)ix / (w - 1);
            float ny = (float)iy / (h - 1);
            net->a[0]->data[0] = nx;
            net->a[0]->data[1] = ny;
            forward(net);
            float out = net->a[net->layers-1]->data[0];
            Uint8 v = (Uint8)(out * 255);
            SDL_SetRenderDrawColor(r, v, v, v, 255);
            SDL_Rect p = {x + ix*pw + 10, y + iy*ph + 10, pw-1, ph-1};
            SDL_RenderFillRect(r, &p);
        }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <image.png>\n"
            "  Image reconstruction demo using xnn.h\n"
            "  Keys: P=pause, R=randomize weights, F=fullscreen view, ←→=upscale (in full view)\n",
            argv[0]);
        return 1;
    }

    img.data = stbi_load(argv[1], &img.w, &img.h, &img.c, 1);
    if (!img.data) {
        fprintf(stderr, "Error: Cannot load image '%s'\n", argv[1]);
        return 1;
    }
    printf("Loaded %s – %dx%d\n", argv[1], img.w, img.h);

    XNN_INIT();

    Network *net  = network_alloc(arch, ARRAY_LEN(arch), acts, LOSS_MSE);
    Network *grad = network_alloc(arch, ARRAY_LEN(arch), acts, LOSS_MSE);
    network_rand(net);

    size_t pixels = img.w * img.h;
    Matrix *in  = matrix_alloc(pixels, 2);
    Matrix *out = matrix_alloc(pixels, 1);

    for (int i = 0; i < img.h; ++i)
        for (int j = 0; j < img.w; ++j) {
            size_t idx = i * img.w + j;
            in->data[idx*2+0] = (float)j / (img.w - 1);
            in->data[idx*2+1] = (float)i / (img.h - 1);
            out->data[idx]    = img.data[idx] / 255.0f;
        }
    Data full = {in, out};

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window *win = SDL_CreateWindow("xnn – Image Reconstruction",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_W, WINDOW_H, SDL_WINDOW_SHOWN);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);

    bool quit = false;
    SDL_Event e;

    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE))
                quit = true;
            if (e.type == SDL_KEYDOWN) {
                switch (e.key.keysym.sym) {
                    case SDLK_p: g_paused = !g_paused; break;
                    case SDLK_r: network_rand(net); g_epoch = 0; g_cost = 1e9f; break;
                    case SDLK_f: g_fullscreen = !g_fullscreen; break;
                    case SDLK_LEFT:  if (g_upscale > 1) g_upscale -= 1; break;
                    case SDLK_RIGHT: g_upscale += 1; break;
                }
            }
        }

        if (!g_paused) {
            for (int i = 0; i < BATCHES_PER_FRAME; ++i) {
                Data batch = { matrix_alloc(BATCH_SIZE, 2), matrix_alloc(BATCH_SIZE, 1) };
                for (int b = 0; b < BATCH_SIZE; ++b) {
                    size_t k = rand() % pixels;
                    batch.in->data[b*2+0] = in->data[k*2+0];
                    batch.in->data[b*2+1] = in->data[k*2+1];
                    batch.out->data[b]    = out->data[k];
                }
                backprop(net, grad, &batch);
                apply_grad(net, grad, LEARNING_RATE / BATCH_SIZE);
                matrix_free(batch.in);
                matrix_free(batch.out);
            }
            g_epoch++;
            if (g_epoch % 40 == 0) g_cost = network_mse(net, &full);
        }

        SDL_SetRenderDrawColor(ren, 9, 9, 9, 255);
        SDL_RenderClear(ren);

        if (!g_fullscreen) {
            render_network(ren, net);
            render_cost(ren);
            render_original(ren, 20, 20 + (int)NET_VIEW_H + 40, (int)VERIFY_SIDE - 40);
            render_prediction(ren, 40 + (int)VERIFY_SIDE, 20 + (int)NET_VIEW_H + 40,
                              (int)VERIFY_SIDE - 40, net, false);
        } else {
            render_prediction(ren, 50, 50, WINDOW_H - 200, net, true);
        }

        SDL_RenderPresent(ren);
    }

    // cleanup
    stbi_image_free(img.data);
    network_free(net); network_free(grad);
    matrix_free(in); matrix_free(out);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
