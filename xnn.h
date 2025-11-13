#ifndef XNN_H_
#define XNN_H_

#include <stdio.h>
#include <stdlib.h>

typedef struct {
	size_t rows;
	size_t cols;
	float *data;

} Matrix;

#endif

#ifdef XNN_IMPLEMENTATION

Matrix *matrix_alloc(size_t rows, size_t cols);
void matrix_fill(Matrix *m, float val);
void matrix_print(Matrix *m);

Matrix *matrix_alloc(size_t rows, size_t cols)
{
	Matrix *m;
	m = malloc(sizeof(Matrix));
	if(!m) return NULL;
	m->rows = rows;
	m->cols = cols;
	m->data = malloc(sizeof(float) * rows * cols);
	if(!m->data) return NULL;

	return m;
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
			printf("%2.2f ", m->data[i * m->cols + j]);
		}
		printf("\n");
	}

	return;
}




#endif  //XNN_IMPLEMENTATION
