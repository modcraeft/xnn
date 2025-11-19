#define XNN_IMPLEMENTATION
#include "xnn.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>

#define RATE               0.25f
#define BATCH_SIZE         32
#define BATCHES_PER_FRAME  600

// Positional encoding: 10 levels → 40 features + original 2 → 42 inputs
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

static void encode_pos(float x, float y, float *out)
{
    int idx = 0;
    out[idx++] = x;
    out[idx++] = y;
    for (int i = 0; i < PE_L; i++) {
        float freq = powf(2.0f, (float)i);
        float ax = freq * x * (float)M_PI;
        float ay = freq * y * (float)M_PI;
        out[idx++] = sinf(ax);
        out[idx++] = cosf(ax);
        out[idx++] = sinf(ay);
        out[idx++] = cosf(ay);
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

void render_network(SDL_Renderer *r, int x, int y, Network *net)
{
    SDL_SetRenderDrawColor(r, 0x55, 0x55, 0x55, 255);
    SDL_Rect border = {x, y, 1110, 820};
    SDL_RenderDrawRect(r, &border);

    const int layers = 5;
    const float layer_width = (1110.0f - 200) / (layers - 1);
    const float x_offset = layer_width / 2.0f;

    for (int l = 0; l < layers; l++) {
        float cx = x + l * layer_width + x_offset;

        int show_neurons = (l == 0) ? 16 :          // show 16 of the 42 encoded inputs
                           (l == 4) ? 1 :           // single output neuron
                           32;                      // hidden layers

        float neuron_h = 820.0f / (float)show_neurons;
        float y_off = neuron_h / 2.0f;

        for (int v = 0; v < show_neurons; v++) {
            size_t real_n = (l == 0) ? (size_t)v * (INPUT_DIM / 16) :
                            (l == 4) ? 0 :
                            (size_t)v * (arch[l] / show_neurons);

            float cy = y + v * neuron_h + y_off;

            // Draw connections
            if (l < layers - 1) {
                int next_show = (l + 1 == 4) ? 1 : 32;
                float next_h = 820.0f / (float)next_show;

                for (int tv = 0; tv < next_show; tv++) {
                    size_t real_t = (l + 1 == 4) ? 0 :
                                    (size_t)tv * (arch[l+1] / next_show);

                    float weight = net->w[l]->data[real_t * arch[l] + real_n];
                    float val = 0.5f + weight * 10.0f;
                    val = val < 0.0f ? 0.0f : val > 1.0f ? 1.0f : val;
                    Uint8 col = (Uint8)(val * 155 + 100);

                    float tx = x + (l + 1) * layer_width + x_offset;
                    float ty = y + tv * next_h + next_h / 2.0f;

                    SDL_SetRenderDrawColor(r, col, col, col, 220);
                    SDL_RenderDrawLine(r, (int)cx, (int)cy, (int)tx, (int)ty);
                }
            }

            // Neuron circle + color
            float bias = (l == 0) ? 0.0f : net->b[l-1]->data[real_n];
            float act = 1.0f / (1.0f + expf(-bias * 5.0f));  // manual sigmoid

            SDL_Color color;
            int radius;

            if (l == 0) {  // encoded inputs
                color = (SDL_Color){0x00, 0xCC, 0xFF, 255};
                radius = 10;
            } else if (l == 4) {  // output
                color = (SDL_Color){0xFF, 0xFF, 0x88, 255};
                radius = 22;
            } else {  // hidden
                Uint8 intensity = (Uint8)(act * 200 + 55);
                color = act > 0.5f ?
                    (SDL_Color){0x88, intensity, 0xFF, 255} :
                    (SDL_Color){intensity, 0x88, 0x88, 255};
                radius = 11;
            }

            SDL_RenderFillCircle(r, (int)cx, (int)cy, radius, color);
        }
    }
}


/*
void render_network(SDL_Renderer *r, int x, int y, Network *net)  // ← pointer!
{
    SDL_SetRenderDrawColor(r, 0x55, 0x55, 0x55, 255);
    SDL_Rect box = {x, y, 1110, 820};
    SDL_RenderDrawRect(r, &box);

    int layers = 5;
    float layer_w = 1110.0f / (layers - 1);

    for (int l = 0; l < layers; l++) {
        float cx = x + l * layer_w;
        size_t neurons = arch[l];
        float nh = 820.0f / (l == 0 ? 16 : neurons);  // show only 16 inputs for clarity

        for (size_t ni = 0; ni < (l == 0 ? 16 : neurons > 16 ? 16 : neurons); ni++) {
            size_t n = (l == 0) ? ni : ni * neurons / 16;
            float cy = y + ni * nh + nh/2;

            if (l < layers-1) {
                size_t samples = arch[l+1] > 16 ? 16 : arch[l+1];
                for (size_t ti = 0; ti < samples; ti++) {
                    size_t t = ti * arch[l+1] / samples;
                    float tx = x + (l+1) * layer_w;
                    float ty = y + ti * 820.0f / samples + 410.0f / samples;
                    float w = net->w[l]->data[t * arch[l] + n];
                    float v = w * 8.0f; v = v < -1 ? -1 : v > 1 ? 1 : v;
                    Uint8 col = (v > 0) ? (Uint8)(v*180 + 75) : 75;
                    Uint8 red = (v < 0) ? (Uint8)(-v*180 + 75) : 75;
                    SDL_SetRenderDrawColor(r, red, col, col, 200);
                    SDL_RenderDrawLine(r, (int)cx, (int)cy, (int)tx, (int)ty);
                }
            }

            int rad = (l == 0) ? 5 : (l == 4) ? 12 : 10;
            SDL_Color c = (l == 0) ? (SDL_Color){0x00,0xAA,0xFF,255} :
                          (l == 4) ? (SDL_Color){0xFF,0xFF,0x00,255} : (SDL_Color){0xFF,0xFF,0xFF,255};
            SDL_RenderFillCircle(r, (int)cx, (int)cy, rad, c);
        }
    }
}
*/
void render_cost(SDL_Renderer *r, int x, int y, float cost, size_t epoch)
{
    SDL_SetRenderDrawColor(r, 0x55, 0x55, 0x55, 255);
    SDL_Rect box = {x, y, 600, 820}; SDL_RenderDrawRect(r, &box);
    int h = (int)(cost * 800); if (h > 800) h = 800;
    SDL_SetRenderDrawColor(r, 255, 80, 80, 160);
    SDL_Rect bar = {x + 60, y + 820 - h, 480, h}; SDL_RenderFillRect(r, &bar);
    render_label(r, x + 70, y + 400, cost, 6, 6);
    render_label(r, x + 70, y + 20, (float)epoch, 0, 5);
}

void render_image(SDL_Renderer *r, int x, int y, int sz)
{
    SDL_SetRenderDrawColor(r, 0x55, 0x55, 0x55, 255);
    SDL_Rect b = {x, y, sz, sz}; SDL_RenderDrawRect(r, &b);
    int pw = sz / td.w, ph = sz / td.h;
    for (int iy = 0; iy < td.h; ++iy)
        for (int ix = 0; ix < td.w; ++ix) {
            Uint8 v = td.img[iy * td.w + ix];
            SDL_SetRenderDrawColor(r, v, v, v, 255);
            SDL_Rect p = {x + ix*pw + 8, y + iy*ph + 8, pw-2, ph-2};
            SDL_RenderFillRect(r, &p);
        }
}

void render_prediction(SDL_Renderer *r, int x, int y, int sz, Network *net, bool hi)
{
    int scale = hi ? (int)up : 1;
    int w = td.w * scale, h = td.h * scale;
    int pw = sz / w, ph = sz / h;

    SDL_Rect b = {x, y, sz, sz};
    SDL_SetRenderDrawColor(r, 0x55, 0x55, 0x55, 255);
    SDL_RenderDrawRect(r, &b);

    float encoded[INPUT_DIM];
    for (int iy = 0; iy < h; ++iy)
        for (int ix = 0; ix < w; ++ix) {
            float nx = (float)ix / (w - 1);
            float ny = (float)iy / (h - 1);
            encode_pos(nx, ny, encoded);
            memcpy(net->a[0]->data, encoded, INPUT_DIM * sizeof(float));
            forward(net);
            float out = net->a[4]->data[0];          // tanh → -1..1
            out = (out + 1.0f) * 0.5f;                // → 0..1
            Uint8 v = (Uint8)(out * 255);
            SDL_SetRenderDrawColor(r, v, v, v, 255);
            SDL_Rect p = {x + ix*pw + 8, y + iy*ph + 8, pw-2, ph-2};
            SDL_RenderFillRect(r, &p);
        }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <image.png>\n", argv[0]);
        return 1;
    }
    td.img = stbi_load(argv[1], &td.w, &td.h, &td.c, 1);
    if (!td.img) { fprintf(stderr, "Failed to load %s\n", argv[1]); return 1; }

    printf("Loaded %dx%d → Positional encoding L=%d → %d inputs\n",
           td.w, td.h, PE_L, INPUT_DIM);

    XNN_INIT();

    Network *net  = network_alloc(arch, 5, act, LOSS_MSE);
    Network *grad = network_alloc(arch, 5, act, LOSS_MSE);
    network_rand(net);

    size_t pixels = (size_t)td.w * td.h;
    Matrix *in  = matrix_alloc(pixels, INPUT_DIM);
    Matrix *out = matrix_alloc(pixels, 1);

    for (int i = 0; i < td.h; ++i)
        for (int j = 0; j < td.w; ++j) {
            size_t idx = (size_t)i * td.w + j;
            float x = (float)j / (td.w - 1);
            float y = (float)i / (td.h - 1);
            encode_pos(x, y, &in->data[idx * INPUT_DIM]);
            out->data[idx] = td.img[idx] / 255.0f;
        }

    Matrix *batch_in  = matrix_alloc(BATCH_SIZE, INPUT_DIM);
    Matrix *batch_out = matrix_alloc(BATCH_SIZE, 1);

    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window   *win = SDL_CreateWindow("xnn – Fourier Features", 50, 50, 1766, 1405, SDL_WINDOW_SHOWN);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
    SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);

    bool quit = false;
    SDL_Event e;

    while (!quit) {
        while (SDL_PollEvent(&e))
            if (e.type == SDL_QUIT || (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE)) quit = true;
            else if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_p) paused = !paused;
                if (e.key.keysym.sym == SDLK_r) { network_rand(net); epochs = 0; cost_store = 1e9f; }
                if (e.key.keysym.sym == SDLK_f) full = !full;
                if (e.key.keysym.sym == SDLK_LEFT  && up > 1) up -= 1;
                if (e.key.keysym.sym == SDLK_RIGHT) up += 1;
            }

        if (!paused) {
            for (int i = 0; i < BATCHES_PER_FRAME; ++i) {
                for (int b = 0; b < BATCH_SIZE; ++b) {
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
            render_network(ren, 20, 20, net);
            render_cost(ren, 1150, 20, cost_store, epochs);
            render_image(ren, 20, 900, 400);
            render_prediction(ren, 450, 900, 400, net, false);
        } else {
            render_prediction(ren, 50, 50, 1300, net, true);
        }

        SDL_RenderPresent(ren);
    }

    stbi_image_free(td.img);
    network_free(net); network_free(grad);
    matrix_free(in); matrix_free(out);
    matrix_free(batch_in); matrix_free(batch_out);
    SDL_DestroyRenderer(ren); SDL_DestroyWindow(win); SDL_Quit();
    return 0;
}
