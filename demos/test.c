#define XNN_IMPLEMENTATION
#include "xnn.h"
#include <assert.h>
#include <stdio.h>

static int predict(Network *net, const float *input)
{
    memcpy(net->a[0]->data, input, net->a[0]->rows * sizeof(float));
    forward(net);
    return net->a[net->layers - 1]->data[0] > 0.5f ? 1 : 0;
}

static void test_matrix(void)
{
    Matrix *a = matrix_alloc(2, 2);
    Matrix *b = matrix_alloc(2, 2);
    Matrix *dst = matrix_alloc(2, 2);
    a->data[0] = 1; a->data[1] = 2; a->data[2] = 3; a->data[3] = 4;
    b->data[0] = 5; b->data[1] = 6; b->data[2] = 7; b->data[3] = 8;
    matrix_dot(dst, a, b);
    assert(dst->data[0] == 19 && dst->data[1] == 22 && dst->data[2] == 43 && dst->data[3] == 50);
    matrix_sum(dst, a);
    assert(dst->data[0] == 20 && dst->data[1] == 24 && dst->data[2] == 46 && dst->data[3] == 54);
    assert(fabsf(matrix_norm(a) - sqrtf(30.0f)) < 1e-6f);
    matrix_free(a); matrix_free(b); matrix_free(dst);
    printf("Matrix tests passed!\n");
}

static void test_xor(void)
{
    size_t arch[] = {2, 12, 12, 1};
    int    act[]  = {ACT_RELU, ACT_RELU, ACT_RELU, ACT_SIGMOID};
    Network *net  = network_alloc(arch, 4, act, LOSS_MSE);
    Network *grad = network_alloc(arch, 4, act, LOSS_MSE);
    assert(net && grad);

    Matrix *in  = matrix_alloc(4, 2);
    Matrix *out = matrix_alloc(4, 1);
    in->data[0] = 0; in->data[1] = 0; out->data[0] = 0;
    in->data[2] = 0; in->data[3] = 1; out->data[1] = 1;
    in->data[4] = 1; in->data[5] = 0; out->data[2] = 1;
    in->data[6] = 1; in->data[7] = 1; out->data[3] = 0;
    Data data = {in, out};

    float init_loss = network_mse(net, &data);
    assert(init_loss > 0.1f);

    for (int epoch = 0; epoch < 10000; ++epoch) {
        backprop(net, grad, &data);
        apply_grad(net, grad, 0.01f);
    }

    float final_loss = network_mse(net, &data);
    printf("Final MSE: %.6f\n", final_loss);
    assert(final_loss < 0.05f);

    float inputs[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    int   expect[4]    = {0,1,1,0};
    for (int i = 0; i < 4; ++i)
        assert(predict(net, inputs[i]) == expect[i]);

    network_free(net); network_free(grad);
    matrix_free(in); matrix_free(out);
    printf("XOR training passed!\n");
}

static void test_grad_check(void)
{
    size_t arch[] = {2, 2, 1};
    int    act[]  = {ACT_RELU, ACT_RELU, ACT_SIGMOID};
    Network *net  = network_alloc(arch, 3, act, LOSS_MSE);
    Network *grad = network_alloc(arch, 3, act, LOSS_MSE);
    assert(net && grad);

    Matrix *in  = matrix_alloc(2, 2);
    Matrix *out = matrix_alloc(2, 1);
    in->data[0] = 0; in->data[1] = 0; out->data[0] = 0;
    in->data[2] = 1; in->data[3] = 1; out->data[1] = 0;
    Data data = {in, out};

    backprop(net, grad, &data);

    const float eps = 1e-4f;
    float max_diff = 0.0f;

    for (size_t l = 0; l < net->layers - 1; ++l) {
        for (size_t j = 0; j < net->w[l]->rows; ++j) {
            for (size_t k = 0; k < net->w[l]->cols; ++k) {
                float orig = net->w[l]->data[j*net->w[l]->cols + k];
                net->w[l]->data[j*net->w[l]->cols + k] = orig + eps;
                float plus  = network_mse(net, &data);
                net->w[l]->data[j*net->w[l]->cols + k] = orig - eps;
                float minus = network_mse(net, &data);
                float num   = (plus - minus) / (2.0f * eps);
                float anal  = grad->w[l]->data[j*grad->w[l]->cols + k];
                float diff  = fabsf(num - anal);
                if (diff > max_diff) max_diff = diff;
                net->w[l]->data[j*net->w[l]->cols + k] = orig;
            }
        }
        for (size_t j = 0; j < net->b[l]->rows; ++j) {
            float orig = net->b[l]->data[j];
            net->b[l]->data[j] = orig + eps;
            float plus  = network_mse(net, &data);
            net->b[l]->data[j] = orig - eps;
            float minus = network_mse(net, &data);
            float num   = (plus - minus) / (2.0f * eps);
            float anal  = grad->b[l]->data[j];
            float diff  = fabsf(num - anal);
            if (diff > max_diff) max_diff = diff;
            net->b[l]->data[j] = orig;
        }
    }

    printf("Gradient check: Max diff = %.6f\n", max_diff);
    assert(max_diff < 5e-2f);

    network_free(net); network_free(grad);
    matrix_free(in); matrix_free(out);
}

int main(void)
{
    XNN_INIT();
    test_matrix();
    test_xor();
    test_grad_check();
    printf("ALL TESTS PASSED – xnn.h is 100%% verified!\n");
    return 0;
}
