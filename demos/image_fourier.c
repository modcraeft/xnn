#define XNN_IMPLEMENTATION
#include "xnn.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define WINDOW_W           1200
#define WINDOW_H           1000

#define RATE               0.25f
#define BATCH_SIZE         32
#define BATCHES_PER_FRAME  600

#define PE_L               10
#define INPUT_DIM          (2 + 4 * PE_L)

size_t arch[] = { INPUT_DIM, 28, 28, 28, 1 };
int    act[]  = { ACT_RELU, ACT_RELU, ACT_RELU, ACT_TANH };

typedef struct { uint8_t *img; int w, h, c; } Img;
Img td = {0};

float  cost_store = 1e9f;
size_t epochs     = 0;
bool   paused     = false;
bool   full       = false;
float  up         = 1.0f;

#include "glyphs.h"

// ==================== SCALABLE LAYOUT ====================
#define NET_X              20
#define NET_Y              20
#define NET_W              (WINDOW_W * 0.63f)      // ~1110
#define NET_H              (WINDOW_H * 0.58f)      // ~820

#define COST_X             (NET_X + NET_W + 20)
#define COST_Y             20
#define COST_W             (WINDOW_W - COST_X - 20)
#define COST_H             NET_H

#define PREVIEW_SIDE       ((WINDOW_H - NET_H - 80) * 0.95f)  // square previews
#define PREVIEW_X          20
#define PREVIEW_Y          (NET_Y + NET_H + 40)
#define PRED_X             (PREVIEW_X + PREVIEW_SIDE + 40)
#define PRED_Y             PREVIEW_Y

// ==================== HELPERS ====================
static void encode_pos(float x, float y, float *out)
{
    int idx = 0;
    out[idx++] = x; out[idx++] = y;
    for (int i = 0; i < PE_L; i++) {
        float f = powf(2.0f, (float)i) * (float)M_PI;
        out[idx++] = sinf(f * x); out[idx++] = cosf(f * x);
        out[idx++] = sinf(f * y); out[idx++] = cosf(f * y);
    }
}

void SDL_RenderFillCircle(SDL_Renderer *r, int cx, int cy, int rad, SDL_Color c)
{
    SDL_SetRenderDrawColor(r, c.r, c.g, c.b, c.a);
    for (int dy = -rad; dy <= rad; dy++)
        for (int dx = -rad; dx <= rad; dx++)
            if (dx*dx + dy*dy <= rad*rad)
                SDL_RenderDrawPoint(r, cx + dx, cy + dy);
}

void render_label(SDL_Renderer *r, int x, int y, float val, int digits, int size)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", digits, val);
    int cell = size;
    for (size_t i = 0; buf[i]; i++)
        for (int gy = 0; gy < 9; gy++)
            for (int gx = 0; gx < 9; gx++)
                if (glyphs[(unsigned char)buf[i]][gy][gx])
                    SDL_RenderFillCircle(r, x + gx*cell + i*81, y + gy*cell + cell/2, 2,
                                         (SDL_Color){0x88,0xBB,0xFF,0xFF});
}

// ==================== SCALABLE RENDER FUNCTIONS ====================
void render_network(SDL_Renderer *r, Network *net)
{
    SDL_SetRenderDrawColor(r, 0x55, 0x55, 0x55, 255);
    SDL_Rect b = {NET_X, NET_Y, (int)NET_W, (int)NET_H}; SDL_RenderDrawRect(r, &b);

    const int layers = 5;
    const float layer_w = (NET_W - 100) / (layers - 1);
    const float x_off = layer_w / 2.0f;

    for (int l = 0; l < layers; l++) {
        float cx = NET_X + l * layer_w + x_off;
        int show = (l == 0) ? 16 : (l == 4) ? 1 : 32;
        float nh = NET_H / (float)show;
        float y_off = nh / 2.0f;

        for (size_t v = 0; v < (size_t)show; v++) {
            size_t n = (l == 0) ? v * (INPUT_DIM / 16) :
                       (l == 4) ? 0 :
                       v * (arch[l] / show);

            float cy = NET_Y + v * nh + y_off;

            if (l < 4) {
                int next_show = (l + 1 == 4) ? 1 : 32;
                float next_nh = NET_H / (float)next_show;

                for (int tv = 0; tv < next_show; tv++) {
                    size_t t = (l + 1 == 4) ? 0 : tv * (arch[l+1] / next_show);
                    float wval = net->w[l]->data[t * arch[l] + n];
                    float val = 0.5f + wval * 10.0f;
                    val = val < 0 ? 0 : val > 1 ? 1 : val;
                    Uint8 col = (Uint8)(val * 155 + 100);

                    float tx = NET_X + (l + 1) * layer_w + x_off;
                    float ty = NET_Y + tv * next_nh + next_nh / 2.0f;

                    SDL_SetRenderDrawColor(r, col, col, col, col/4 + 10);
                    SDL_RenderDrawLine(r, (int)cx, (int)cy, (int)tx, (int)ty);
                }
            }

            float bias = (l == 0) ? 0.0f : net->b[l-1]->data[n];
            float act = 1.0f / (1.0f + expf(-bias * 5.0f));

            SDL_Color col;
            int rad;
            if (l == 0) { col = (SDL_Color){0x00,0xCC,0xFF,255}; rad = 10; }
            else if (l == 4) { col = (SDL_Color){0xFF,0xFF,0x88,255}; rad = 22; }
            else {
                Uint8 i = (Uint8)(act * 200 + 55);
                col = act > 0.5f ? (SDL_Color){0x88,i,0xFF,255} : (SDL_Color){i,0x88,0x88,255};
                rad = 11;
            }
            SDL_RenderFillCircle(r, (int)cx, (int)cy, rad, col);
        }
    }
}

void render_cost(SDL_Renderer *r)
{
    SDL_SetRenderDrawColor(r, 0x55, 0x55, 0x55, 255);
    SDL_Rect b = {COST_X, COST_Y, (int)COST_W, (int)COST_H}; SDL_RenderDrawRect(r, &b);

    int h = (int)(cost_store * COST_H * 0.9f);
    if (h > (int)COST_H) h = (int)COST_H;
    SDL_SetRenderDrawColor(r, 255, 80, 80, 160);
    SDL_Rect bar = {COST_X + 60, COST_Y + (int)COST_H - h, (int)COST_W - 120, h};
    SDL_RenderFillRect(r, &bar);

    render_label(r, COST_X + 70, COST_Y + 400, cost_store, 6, 6);
    render_label(r, COST_X + 70, COST_Y + 20, (float)epochs, 0, 5);
}

void render_image(SDL_Renderer *r)
{
    SDL_SetRenderDrawColor(r, 0x55, 0x55, 0x55, 255);
    SDL_Rect b = {PREVIEW_X, PREVIEW_Y, (int)PREVIEW_SIDE, (int)PREVIEW_SIDE}; SDL_RenderDrawRect(r, &b);

    int pw = (int)PREVIEW_SIDE / td.w;
    int ph = (int)PREVIEW_SIDE / td.h;
    for (int iy = 0; iy < td.h; iy++)
        for (int ix = 0; ix < td.w; ix++) {
            Uint8 v = td.img[iy * td.w + ix];
            SDL_SetRenderDrawColor(r, v, v, v, 255);
            SDL_Rect p = {PREVIEW_X + ix*pw + 8, PREVIEW_Y + iy*ph + 8, pw-2, ph-2};
            SDL_RenderFillRect(r, &p);
        }
}

void render_prediction(SDL_Renderer *r, Network *net, bool hi)
{
    int scale = hi ? (int)up : 1;
    int w = td.w * scale, h = td.h * scale;
    int sz = hi ? (WINDOW_H - 160) : (int)PREVIEW_SIDE;
    int pw = sz / w, ph = sz / h;
    int px = hi ? 50 : PRED_X;
    int py = hi ? 50 : PRED_Y;

    SDL_SetRenderDrawColor(r, 0x55, 0x55, 0x55, 255);
    SDL_Rect b = {px, py, sz, sz}; SDL_RenderDrawRect(r, &b);

    float encoded[INPUT_DIM];
    for (int iy = 0; iy < h; iy++)
        for (int ix = 0; ix < w; ix++) {
            float nx = (float)ix / (w - 1);
            float ny = (float)iy / (h - 1);
            encode_pos(nx, ny, encoded);
            memcpy(net->a[0]->data, encoded, INPUT_DIM * sizeof(float));
            forward(net);
            float out = (net->a[4]->data[0] + 1.0f) * 0.5f;
            Uint8 v = (Uint8)(out * 255);
            SDL_SetRenderDrawColor(r, v, v, v, 255);
            SDL_Rect p = {px + ix*pw + 8, py + iy*ph + 8, pw-2, ph-2};
            SDL_RenderFillRect(r, &p);
        }
}

// ==================== MAIN ====================
int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "Usage: %s <image.png>\n", argv[0]); return 1; }
    td.img = stbi_load(argv[1], &td.w, &td.h, &td.c, 1);
    if (!td.img) { fprintf(stderr, "Failed to load %s\n", argv[1]); return 1; }

    printf("Loaded %dx%d → Fourier features (L=%d)\n", td.w, td.h, PE_L);

    XNN_INIT();

    Network *net  = network_alloc(arch, 5, act, LOSS_MSE);
    Network *grad = network_alloc(arch, 5, act, LOSS_MSE);
    network_rand(net);

    size_t pixels = (size_t)td.w * td.h;
    Matrix *in  = matrix_alloc(pixels, INPUT_DIM);
    Matrix *out = matrix_alloc(pixels, 1);

    for (int i = 0; i < td.h; i++)
        for (int j = 0; j < td.w; j++) {
            size_t idx = (size_t)i * td.w + j;
            float x = (float)j / (td.w - 1);
            float y = (float)i / (td.h - 1);
            encode_pos(x, y, &in->data[idx * INPUT_DIM]);
            out->data[idx] = td.img[idx] / 255.0f;
        }

    Matrix *batch_in  = matrix_alloc(BATCH_SIZE, INPUT_DIM);
    Matrix *batch_out = matrix_alloc(BATCH_SIZE, 1);

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window   *win = SDL_CreateWindow("xnn – Fourier Features", 50, 50, WINDOW_W, WINDOW_H, SDL_WINDOW_SHOWN);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);

    bool quit = false;
    SDL_Event e;

    while (!quit) {
        while (SDL_PollEvent(&e))
            if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)) quit = true;
            else if (e.type == SDL_KEYDOWN) {
                if (

e.key.keysym.sym == SDLK_p) paused = !paused;
                if (e.key.keysym.sym == SDLK_r) { network_rand(net); epochs = 0; cost_store = 1e9f; }
                if (e.key.keysym.sym == SDLK_f) full = !full;
                if (e.key.keysym.sym == SDLK_LEFT  && up > 1) up -= 1.0f;
                if (e.key.keysym.sym == SDLK_RIGHT) up += 1.0f;
            }

        if (!paused) {
            for (int i = 0; i < BATCHES_PER_FRAME; i++) {
                for (int b = 0; b < BATCH_SIZE; b++) {
                    size_t k = rand() % pixels;
                    memcpy(&batch_in->data[b * INPUT_DIM], &in->data[k * INPUT_DIM], INPUT_DIM * sizeof(float));
                    batch_out->data[b] = out->data[k];
                }
                backprop(net, grad, &(Data){batch_in, batch_out});
                apply_grad(net, grad, RATE / BATCH_SIZE);
            }
            epochs += BATCHES_PER_FRAME;
            if (epochs % 300 == 0)
                cost_store = network_mse(net, &(Data){in, out});
        }

        SDL_SetRenderDrawColor(ren, 9, 9, 9, 255);
        SDL_RenderClear(ren);

        if (!full) {
            render_network(ren, net);
            render_cost(ren);
            render_image(ren);
            render_prediction(ren, net, false);
        } else {
            render_prediction(ren, net, true);
        }

        SDL_RenderPresent(ren);
    }

    // cleanup
    stbi_image_free(td.img);
    network_free(net); network_free(grad);
    matrix_free(in); matrix_free(out);
    matrix_free(batch_in); matrix_free(batch_out);
    SDL_DestroyRenderer(ren); SDL_DestroyWindow(win); SDL_Quit();
    return 0;
}
