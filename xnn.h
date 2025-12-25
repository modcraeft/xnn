// xnn.h (added ACT_LINEAR support)
#ifndef XNN_H_
#define XNN_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ------------------------------------------------------------------
 * Macros
 * ------------------------------------------------------------------ */
#define ARRAY_LEN(x) ( (sizeof(x)) / (sizeof(x[0])) )

/* ------------------------------------------------------------------
 * Activation & Loss function IDs
 * ------------------------------------------------------------------ */
typedef enum {
    ACT_SIGMOID = 0,
    ACT_TANH    = 1,
    ACT_RELU    = 2,
    ACT_SOFTMAX = 3,
    ACT_LINEAR  = 4  // Added for identity (no activation)
} Activation;

typedef enum {
    LOSS_MSE = 0,
    LOSS_CE  = 1
} Loss;

/* ------------------------------------------------------------------
 * Matrix & Network
 * ------------------------------------------------------------------ */
typedef struct { size_t rows, cols; float *data; } Matrix;
typedef struct Network Network;
typedef struct { Matrix *in, *out; } Data;

/* ------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------ */
Matrix *matrix_alloc(size_t r, size_t c);
void matrix_free(Matrix *m);
float rand_float(float lo, float hi);
void matrix_rand(Matrix *m, float lo, float hi);
void matrix_rand_bias(Matrix *m);
void matrix_fill(Matrix *m, float v);
void matrix_print(const Matrix *m);
float matrix_norm(const Matrix *m);
int matrix_sum(Matrix *dst, const Matrix *src);
int matrix_dot(Matrix *dst, const Matrix *a, const Matrix *b);
int matrix_copy(Matrix *dst, const Matrix *src);

Network *network_alloc(const size_t *arch, size_t n, const int *act, int loss);
void network_free(Network *net);
void network_rand(Network *net);
void network_zero(Network *net);
void network_print(const Network *net);
void forward(Network *net);
void backprop(Network *net, Network *grad, const Data *data);
void apply_grad(Network *net, const Network *grad, float rate);

int network_save(const Network *net, const char *path);
Network *network_load(const char *path, const size_t *arch, size_t n, const int *act, int loss);

Matrix *matrix_from_csv(const char *path, size_t rows, size_t cols);
float network_mse(const Network *net, const Data *data);
void network_predict(const Network *net, const float *input, float *output);

#endif /* XNN_H_ */

/* ==============================================================
 * IMPLEMENTATION
 * ============================================================== */
#ifdef XNN_IMPLEMENTATION

struct Network {
    size_t layers;
    Matrix **w, **b, **a;
    int *activations;
    int loss;
};

/* ---------- Matrix ---------- */
Matrix *matrix_alloc(size_t r, size_t c)
{
    if (!r || !c) return NULL;
    Matrix *m = malloc(sizeof*m);
    if (!m) return NULL;
    m->rows = r; m->cols = c;
    m->data = malloc(r*c*sizeof(float));
    if (!m->data) { free(m); return NULL; }
    return m;
}
void matrix_free(Matrix *m){ if(m){free(m->data);free(m);} }

int matrix_copy(Matrix *dst, const Matrix *src)
{
    if (!dst || !src || dst->rows != src->rows || dst->cols != src->cols) return -1;
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(float));
    return 0;
}
float rand_float(float lo, float hi)
{ return lo + (float)rand()/RAND_MAX * (hi-lo); }
void matrix_rand(Matrix *m, float lo, float hi)
{ for(size_t i=0;i<m->rows*m->cols;i++) m->data[i]=rand_float(lo,hi); }
void matrix_rand_bias(Matrix *m){ matrix_rand(m,-0.1f,0.1f); }
void matrix_fill(Matrix *m, float v)
{ for(size_t i=0;i<m->rows*m->cols;i++) m->data[i]=v; }
void matrix_print(const Matrix *m)
{
    for(size_t i=0;i<m->rows;i++){
        for(size_t j=0;j<m->cols;j++) printf("%.5f ",m->data[i*m->cols+j]);
        printf("\n");
    }
}
float matrix_norm(const Matrix *m)
{
    if (!m) return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < m->rows * m->cols; ++i)
        sum += m->data[i] * m->data[i];
    return sqrtf(sum);
}
int matrix_sum(Matrix *dst, const Matrix *src)
{
    if(!dst||!src||dst->rows!=src->rows||dst->cols!=src->cols) return -1;
    for(size_t i=0;i<dst->rows*dst->cols;i++) dst->data[i]+=src->data[i];
    return 0;
}
int matrix_dot(Matrix *dst, const Matrix *a, const Matrix *b)
{
    if(!dst||!a||!b||a->cols!=b->rows||dst->rows!=a->rows||dst->cols!=b->cols) return -1;
    matrix_fill(dst,0);
    for(size_t i=0;i<a->rows;i++)
        for(size_t j=0;j<b->cols;j++)
            for(size_t k=0;k<a->cols;k++)
                dst->data[i*dst->cols+j] +=
                    a->data[i*a->cols+k] * b->data[k*b->cols+j];
    return 0;
}

/* ---------- Activations ---------- */
static void act_sigmoid(Matrix *m)
{ for(size_t i=0;i<m->rows*m->cols;i++) m->data[i]=1/(1+expf(-m->data[i])); }
static float dact_sigmoid(float x)
{ float s=1/(1+expf(-x)); return s*(1-s); }
static void act_tanh(Matrix *m)
{ for(size_t i=0;i<m->rows*m->cols;i++) m->data[i]=tanhf(m->data[i]); }
static float dact_tanh(float x)
{ float t=tanhf(x); return 1-t*t; }
static void act_relu(Matrix *m)
{ for(size_t i=0;i<m->rows*m->cols;i++) if(m->data[i]<0) m->data[i]=0; }
static float dact_relu(float x) { return x>0?1:0; }
static void act_softmax(Matrix *m)
{
    if (!m || m->cols != 1) return;
    float max = m->data[0];
    for (size_t i = 1; i < m->rows; ++i)
        if (m->data[i] > max) max = m->data[i];
    float sum = 0.0f;
    for (size_t i = 0; i < m->rows; ++i) {
        m->data[i] = expf(m->data[i] - max);
        sum += m->data[i];
    }
    for (size_t i = 0; i < m->rows; ++i) m->data[i] /= sum;
}
static void act_linear(Matrix *m) { /* identity */ }
static float dact_linear(float x) { return 1.0f; }

/* ---------- Network ---------- */
Network *network_alloc(const size_t *arch, size_t n, const int *act, int loss)
{
    if(!arch||n<2||!act) return NULL;
    Network *net = malloc(sizeof*net);
    if(!net) return NULL;
    net->layers = n;
    net->loss = loss;
    net->activations = malloc(sizeof(int)*n);
    if(!net->activations) { free(net); return NULL; }
    memcpy(net->activations, act, sizeof(int)*n);
    net->w = malloc(sizeof(Matrix*)*(n-1));
    net->b = malloc(sizeof(Matrix*)*(n-1));
    net->a = malloc(sizeof(Matrix*)*n);
    if(!net->w||!net->b||!net->a) goto fail;
    net->a[0] = matrix_alloc(arch[0],1);
    if(!net->a[0]) goto fail;
    for(size_t i=1;i<n;i++){
        net->w[i-1]=matrix_alloc(arch[i],arch[i-1]);
        net->b[i-1]=matrix_alloc(arch[i],1);
        net->a[i] =matrix_alloc(arch[i],1);
        if(!net->w[i-1]||!net->b[i-1]||!net->a[i]) goto fail;
    }
    network_rand(net);
    return net;
fail:
    network_free(net);
    return NULL;
}
void network_free(Network *net)
{
    if(!net) return;
    if(net->activations) free(net->activations);
    if(net->a){ for(size_t i=0;i<net->layers;i++) matrix_free(net->a[i]); free(net->a); }
    if(net->w){ for(size_t i=0;i<net->layers-1;i++) matrix_free(net->w[i]); free(net->w); }
    if(net->b){ for(size_t i=0;i<net->layers-1;i++) matrix_free(net->b[i]); free(net->b); }
    free(net);
}
void network_rand(Network *net)
{
    if(!net) return;
    for(size_t i=0;i<net->layers-1;i++){
        Matrix *w = net->w[i];
        size_t fan_in = w->cols, fan_out = w->rows;
        float limit;
        int act_type = net->activations[i+1];
        if (act_type == ACT_RELU || act_type == ACT_LINEAR) {
            limit = sqrtf(2.0f / fan_in);
        } else {
            limit = sqrtf(6.0f / (fan_in + fan_out));
        }
        for(size_t j=0;j<w->rows*w->cols;j++)
            w->data[j] = rand_float(-limit, limit);
        if(net->b[i]) matrix_rand_bias(net->b[i]);
    }
}
void network_zero(Network *net)
{
    if(!net) return;
    for(size_t i=0;i<net->layers-1;i++){
        matrix_fill(net->w[i],0);
        matrix_fill(net->b[i],0);
    }
}
void network_print(const Network *net)
{
    if (!net) return;
    printf("Network: %zu layers, loss=%s\n", net->layers,
           net->loss == LOSS_MSE ? "MSE" : "Cross-Entropy");
    for (size_t i = 0; i < net->layers - 1; ++i) {
        const char* act_str = 
            net->activations[i+1] == ACT_SIGMOID ? "Sigmoid" :
            net->activations[i+1] == ACT_TANH ? "Tanh" :
            net->activations[i+1] == ACT_RELU ? "ReLU" :
            net->activations[i+1] == ACT_SOFTMAX ? "Softmax" :
            net->activations[i+1] == ACT_LINEAR ? "Linear" : "Unknown";
        printf("Layer %zu -> %zu: %s\n", i, i+1, act_str);
        printf(" W: "); matrix_print(net->w[i]);
        printf(" b: "); matrix_print(net->b[i]);
    }
}
void forward(Network *net)
{
    if(!net) return;
    for(size_t i=0;i<net->layers-1;i++){
        matrix_dot(net->a[i+1], net->w[i], net->a[i]);
        matrix_sum(net->a[i+1], net->b[i]);
        int act = net->activations[i+1];
        if(act == ACT_SIGMOID) act_sigmoid(net->a[i+1]);
        else if(act == ACT_TANH) act_tanh(net->a[i+1]);
        else if(act == ACT_RELU) act_relu(net->a[i+1]);
        else if(act == ACT_SOFTMAX) act_softmax(net->a[i+1]);
        else if(act == ACT_LINEAR) act_linear(net->a[i+1]);
    }
}
void backprop(Network *net, Network *grad, const Data *data)
{
    if(!net||!grad||!data||!data->in||!data->out) return;
    size_t batch = data->in->rows;
    size_t in_sz = data->in->cols;
    size_t out_sz= data->out->cols;
    size_t L = net->layers-1;
    if(in_sz!=net->a[0]->rows || out_sz!=net->a[L]->rows || batch!=data->out->rows) return;

    network_zero(grad);

    for(size_t s=0;s<batch;s++){
        for(size_t l=1;l<net->layers;l++) matrix_fill(grad->a[l], 0);

        memcpy(net->a[0]->data, &data->in->data[s*in_sz], in_sz*sizeof(float));
        forward(net);

        for(size_t j=0;j<out_sz;j++){
            float p = net->a[L]->data[j];
            float t = data->out->data[s*out_sz+j];
            if (net->loss == LOSS_MSE) {
                grad->a[L]->data[j] = 2*(p-t);
            } else {
                grad->a[L]->data[j] = p - t;
            }
        }

        for(size_t l=L;l>0;--l){
            for(size_t j=0;j<net->a[l]->rows;j++){
                float a  = net->a[l]->data[j];
                float da = grad->a[l]->data[j];
                int act  = net->activations[l];
                float ds;
                if (l == L && net->loss == LOSS_CE && act == ACT_SOFTMAX) {
                    ds = 1.0f;
                } else {
                    ds = (act == ACT_SIGMOID) ? dact_sigmoid(a) :
                         (act == ACT_TANH)    ? dact_tanh(a)    :
                         (act == ACT_RELU)    ? dact_relu(a)    :
                         (act == ACT_LINEAR)  ? dact_linear(a)  : 0;
                }
                grad->b[l-1]->data[j] += da * ds;
                for(size_t k=0;k<net->a[l-1]->rows;k++){
                    float prev = net->a[l-1]->data[k];
                    float w    = net->w[l-1]->data[j*net->w[l-1]->cols + k];
                    grad->w[l-1]->data[j*net->w[l-1]->cols + k] += da * ds * prev;
                    if(l>1) grad->a[l-1]->data[k] += da * ds * w;
                }
            }
        }
    }

    for(size_t i=0;i<grad->layers-1;i++){
        size_t n = grad->w[i]->rows*grad->w[i]->cols;
        for(size_t j=0;j<n;j++) grad->w[i]->data[j] /= (float)batch;
        n = grad->b[i]->rows;
        for(size_t j=0;j<n;j++) grad->b[i]->data[j] /= (float)batch;
    }
}
void apply_grad(Network *net, const Network *grad, float rate)
{
    if(!net||!grad) return;
    for(size_t i=0;i<net->layers-1;i++){
        size_t n = net->w[i]->rows*net->w[i]->cols;
        for(size_t j=0;j<n;j++) net->w[i]->data[j] -= grad->w[i]->data[j]*rate;
        n = net->b[i]->rows;
        for(size_t j=0;j<n;j++) net->b[i]->data[j] -= grad->b[i]->data[j]*rate;
    }
}

/* ---------- Save / Load ---------- */
int network_save(const Network *net, const char *path)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fwrite(&net->layers, sizeof(size_t), 1, f);
    fwrite(&net->loss, sizeof(int), 1, f);
    for (size_t i = 0; i < net->layers - 1; ++i) {
        size_t w_sz = net->w[i]->rows * net->w[i]->cols;
        size_t b_sz = net->b[i]->rows;
        fwrite(net->w[i]->data, sizeof(float), w_sz, f);
        fwrite(net->b[i]->data, sizeof(float), b_sz, f);
    }
    fclose(f);
    return 0;
}

Network *network_load(const char *path, const size_t *arch, size_t n, const int *act, int loss)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    size_t layers;
    int saved_loss;
    if (fread(&layers, sizeof(size_t), 1, f) != 1 || layers != n) {
        fclose(f);
        return NULL;
    }
    if (fread(&saved_loss, sizeof(int), 1, f) != 1 || saved_loss != loss) {
        fclose(f);
        return NULL;
    }

    Network *net = network_alloc(arch, n, act, loss);
    if (!net) {
        fclose(f);
        return NULL;
    }

    for (size_t i = 0; i < n - 1; ++i) {
        size_t w_sz = net->w[i]->rows * net->w[i]->cols;
        size_t b_sz = net->b[i]->rows;
        if (fread(net->w[i]->data, sizeof(float), w_sz, f) != w_sz ||
            fread(net->b[i]->data, sizeof(float), b_sz, f) != b_sz) {
            network_free(net);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    return net;
}

/* ---------- Utilities ---------- */
Matrix *matrix_from_csv(const char *path, size_t rows, size_t cols)
{
    FILE *f = fopen(path, "r");
    if (!f) return NULL;
    Matrix *m = matrix_alloc(rows, cols);
    if (!m) { fclose(f); return NULL; }
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            if (fscanf(f, "%f,", &m->data[i*cols + j]) != 1)
                goto fail;
    fclose(f);
    return m;
fail:
    matrix_free(m);
    fclose(f);
    return NULL;
}

float network_mse(const Network *net, const Data *data)
{
    if (!net || !data || !data->in || !data->out) return 0.0f;
    size_t batch = data->in->rows;
    size_t in_sz = data->in->cols;
    size_t out_sz = data->out->cols;
    size_t L = net->layers - 1;
    if (in_sz != net->a[0]->rows || out_sz != net->a[L]->rows) return 0.0f;

    /* Clone the network so we don't mutate the original */
    size_t *tmp_arch = malloc(net->layers * sizeof(size_t));
    int *tmp_act = malloc(net->layers * sizeof(int));
    for (size_t i = 0; i < net->layers; ++i) {
        tmp_arch[i] = net->a[i]->rows;
        tmp_act[i]  = net->activations[i];
    }
    Network *tmp = network_alloc(tmp_arch, net->layers, tmp_act, net->loss);
    free(tmp_arch); free(tmp_act);
    if (!tmp) return 0.0f;
    for (size_t i = 0; i < net->layers - 1; ++i) {
        matrix_copy(tmp->w[i], net->w[i]);
        matrix_copy(tmp->b[i], net->b[i]);
    }

    float loss = 0.0f;
    for (size_t s = 0; s < batch; ++s) {
        memcpy(tmp->a[0]->data, &data->in->data[s * in_sz], in_sz * sizeof(float));
        forward(tmp);
        for (size_t j = 0; j < out_sz; ++j) {
            float d = tmp->a[L]->data[j] - data->out->data[s * out_sz + j];
            loss += d * d;
        }
    }
    network_free(tmp);
    return loss / batch;
}

void network_predict(const Network *net, const float *input, float *output)
{
    if (!net || !input || !output) return;
    memcpy(net->a[0]->data, input, net->a[0]->rows * sizeof(float));
    forward((Network*)net);
    memcpy(output, net->a[net->layers - 1]->data, net->a[net->layers - 1]->rows * sizeof(float));
}

/* Seed RNG once */
static void init_xnn(void)
{
    static int done = 0;
    if (!done) {
        srand((unsigned)time(NULL));
        done = 1;
    }
}
#define XNN_INIT() init_xnn()

#endif /* XNN_IMPLEMENTATION */
