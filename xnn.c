#define XNN_IMPLEMENTATION
#include "xnn.h"


int main()
{
	init_xnn();

	Matrix *m = matrix_alloc(20,20);
	if(!m) {
		fprintf(stderr, "ERROR: Matrix allocation Failed");
		return -1;
	}

	//matrix_fill(m, 0.0f);
	matrix_rand(m, 1.0f, 0.0f);
	matrix_print(m);


	return 0;
}
