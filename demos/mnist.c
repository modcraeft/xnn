#define XNN_IMPLEMENTATION
#include "xnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>

#define BATCH_SIZE 64
#define LEARNING_RATE 0.1f

static void shuffle(int *idx, int n) {
    for (int i = n-1; i > 0; --i) {
        int j = rand() % (i+1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
}

/*
static float cross_entropy(const float *pred, const float *target, size_t n) {
    float loss = 0.0f;
    for (size_t i = 0; i < n; ++i)
        loss -= target[i] * logf(pred[i] + 1e-12f);
    return loss;
}
*/

static float grad_norm(const Network *grad) {
    float norm = 0.0f;
    for (size_t l = 0; l < grad->layers - 1; ++l) {
        norm += matrix_norm(grad->w[l]) * matrix_norm(grad->w[l]);
        norm += matrix_norm(grad->b[l]) * matrix_norm(grad->b[l]);
    }
    return sqrtf(norm);
}

static void clip_grad(Network *grad, float max_norm) {
    float norm = grad_norm(grad);
    if (norm > max_norm) {
        float scale = max_norm / (norm + 1e-6f);
        for (size_t l = 0; l < grad->layers - 1; ++l) {
            for (size_t i = 0; i < grad->w[l]->rows * grad->w[l]->cols; ++i)
                grad->w[l]->data[i] *= scale;
            for (size_t i = 0; i < grad->b[l]->rows; ++i)
                grad->b[l]->data[i] *= scale;
        }
    }
}

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

int main(void) {
    XNN_INIT();
    srand(time(NULL));

	const char *train_path = "mnist_train.csv";
    const char *test_path  = "mnist_test.csv";

    // Check if files exist
    if (!file_exists(train_path) || !file_exists(test_path)) {
        fprintf(stderr, "\n");
        fprintf(stderr, "ERROR: MNIST CSV files not found!\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "   Required files:\n");
        fprintf(stderr, "     • %s\n", train_path);
        fprintf(stderr, "     • %s\n", test_path);
        fprintf(stderr, "\n");
        fprintf(stderr, "   git clone them from: https://github.com/phoebetronic/mnist");
        fprintf(stderr, "\n");
        return 1;
    }

    printf("Loading MNIST data...\n");
    Matrix *train_raw = matrix_from_csv(train_path, 60000, 785);
    Matrix *test_raw  = matrix_from_csv(test_path,  10000, 785);
    if (!train_raw || !test_raw) {
        fprintf(stderr, "Failed to load CSV. Check format and paths.\n");
        return 1;
    }

    Matrix *X_train = matrix_alloc(60000, 784);
    Matrix *y_train = matrix_alloc(60000, 10);
    Matrix *X_test  = matrix_alloc(10000, 784);
    Matrix *y_test  = matrix_alloc(10000, 10);

    for (int i = 0; i < 60000; ++i) {
        int label = (int)train_raw->data[i*785];
        for (int j = 0; j < 784; ++j)
            X_train->data[i*784 + j] = train_raw->data[i*785 + 1 + j] / 255.0f;
        for (int j = 0; j < 10; ++j) y_train->data[i*10 + j] = (j == label) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < 10000; ++i) {
        int label = (int)test_raw->data[i*785];
        for (int j = 0; j < 784; ++j)
            X_test->data[i*784 + j] = test_raw->data[i*785 + 1 + j] / 255.0f;
        for (int j = 0; j < 10; ++j) y_test->data[i*10 + j] = (j == label) ? 1.0f : 0.0f;
    }

    matrix_free(train_raw); matrix_free(test_raw);

    /* Network */
    size_t arch[] = {784, 128, 10};
    int    act[]  = {ACT_RELU, ACT_RELU, ACT_SOFTMAX};
    Network *net  = network_alloc(arch, 3, act, LOSS_CE);
    Network *grad = network_alloc(arch, 3, act, LOSS_CE);

    /* Mini-batch buffers */
    Matrix *batch_in  = matrix_alloc(BATCH_SIZE, 784);
    Matrix *batch_out = matrix_alloc(BATCH_SIZE, 10);
    Data batch_data = {batch_in, batch_out};

    int *indices = malloc(60000 * sizeof(int));
    for (int i = 0; i < 60000; ++i) indices[i] = i;

    printf("=== MNIST MINI-BATCH TRAINING (batch=%d, lr=%.3f) ===\n", BATCH_SIZE, LEARNING_RATE);
    for (int epoch = 0; epoch < 50; ++epoch) {
        shuffle(indices, 60000);

        for (int b = 0; b < 60000; b += BATCH_SIZE) {
            int bs = (b + BATCH_SIZE > 60000) ? (60000 - b) : BATCH_SIZE;
            batch_in->rows = bs; batch_out->rows = bs;

            for (int i = 0; i < bs; ++i) {
                int idx = indices[b + i];
                memcpy(&batch_in->data[i*784], &X_train->data[idx*784], 784*sizeof(float));
                memcpy(&batch_out->data[i*10],  &y_train->data[idx*10],  10*sizeof(float));
            }

            backprop(net, grad, &batch_data);
            clip_grad(grad, 5.0f);
            apply_grad(net, grad, LEARNING_RATE);
        }

        if (epoch % 5 == 0 || epoch == 49) {
            int correct = 0;
            for (int i = 0; i < 10000; ++i) {
                float out[10];
                network_predict(net, &X_test->data[i*784], out);
                int pred = 0;
                for (int j = 1; j < 10; ++j) if (out[j] > out[pred]) pred = j;
                int true_lbl = -1;
                for (int j = 0; j < 10; ++j)
                    if (y_test->data[i*10 + j] > 0.5f) { true_lbl = j; break; }
                if (pred == true_lbl) ++correct;
            }
            float acc = 100.0f * correct / 10000.0f;
            printf("Epoch %2d | Test Acc: %.3f%%\n", epoch, acc);
        }
    }

    network_save(net, "mnist_model.bin");
    printf("Model saved to mnist_model.bin\n");

    // Cleanup
    matrix_free(X_train); matrix_free(y_train);
    matrix_free(X_test);  matrix_free(y_test);
    matrix_free(batch_in); matrix_free(batch_out);
    free(indices);
    network_free(net); network_free(grad);
    return 0;
}
