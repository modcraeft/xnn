#ifndef XNN_H_
#define XNN_H_

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
	size_t rows;
	size_t cols;
	float *data;

} Matrix;

#endif

#ifdef XNN_IMPLEMENTATION

Matrix *matrix_alloc(size_t rows, size_t cols);
float rand_float(float hi, float low);
void matrix_rand(Matrix *m, float hi, float low);
void matrix_fill(Matrix *m, float val);
void matrix_print(Matrix *m);

void init_xnn()
{
	srand(time(0));

	return;
}

Matrix *matrix_alloc(size_t rows, size_t cols)
{
	if(rows == 0 || cols == 0) return NULL;
	Matrix *m;
	m = malloc(sizeof(Matrix));
	if(!m) return NULL;
	m->rows = rows;
	m->cols = cols;
	m->data = malloc(sizeof(float) * rows * cols);
	if(!m->data) { 
		free(m);
		return NULL; 
	}

	return m;
}

float rand_float(float hi, float low)
{
	return low + ((float)rand() / RAND_MAX) * (hi - low);
}

void matrix_rand(Matrix *m, float hi, float low)
{
	size_t n = m->rows * m->cols;
	for(size_t i = 0; i < n; i++) {	
		m->data[i] = rand_float(hi, low);
	}
	
	return;
}

void matrix_fill(Matrix *m, float val)
{
	size_t n = m->rows * m->cols;
	for(size_t i = 0; i < n; i++) {	
		m->data[i] = val;
	}

	return;
}

void matrix_print(Matrix *m)
{
	for(size_t i = 0; i < m->rows; i++) {
		for(size_t j = 0; j < m->cols; j++) {
			printf("%.2f ", m->data[i * m->cols + j]);
		}
		printf("\n");
	}

	return;
}

int matrix_sum(Matrix *dst, Matrix *a, Matrix *b) {
    // Check if inputs are valid and dimensions match
    if (!dst || !a || !b || 
        a->rows != b->rows || a->cols != b->cols || 
        dst->rows != a->rows || dst->cols != a->cols) {
        return -1; // Error: invalid inputs or incompatible dimensions
    }

    // Perform element-wise addition
    size_t i;
    for (i = 0; i < a->rows * a->cols; i++) {
        dst->data[i] = a->data[i] + b->data[i];
    }

    return 0; // Success
}

int matrix_dot(Matrix *dst, const Matrix *a, const Matrix *b) {
    // Check if inputs are valid and dimensions are compatible
    if (!dst || !a || !b || a->cols != b->rows || 
        dst->rows != a->rows || dst->cols != b->cols) {
        return -1; // Error: invalid inputs or incompatible dimensions
    }

    // Initialize destination matrix to zero
    size_t i, j, k;
    for (i = 0; i < dst->rows * dst->cols; i++) {
        dst->data[i] = 0.0f;
    }

    // Compute dot product
    for (i = 0; i < a->rows; i++) {
        for (j = 0; j < b->cols; j++) {
            for (k = 0; k < a->cols; k++) {
                dst->data[i * dst->cols + j] += 
                    a->data[i * a->cols + k] * b->data[k * b->cols + j];
            }
        }
    }

    return 0; // Success
}



#endif  //XNN_IMPLEMENTATION
