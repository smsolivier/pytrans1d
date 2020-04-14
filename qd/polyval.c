void PolyVal(int N, int p, double* B, double x, double* eval) {
	for (int i=0; i<N; i++) {
		eval[i] = B[i]; 
	}

	for (int n=0; n<N; n++) {
		for (int i=1; i<p; i++) {
			eval[n] = eval[n]*x + B[n + i*N]; 
		}
	}
}