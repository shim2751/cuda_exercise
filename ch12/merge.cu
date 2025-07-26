__global__
void merge_sequential(int* A, int m, int* B, int n, int* C) {
    int i = 0, j = 0, k = 0;
    while(i < m && j < n){
        if(A[i] < B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if(i == m){
        while(j < n) {
            C[k++] = B[j++];
        }
    } else {
        while(i < m) {
            C[k++] = A[i++];
        }
    }
}

__global__
int co_rank(int k, int* A, int m, int* B, int n){
    int i = k < m ? k : m; // min(k, m)  k가 m보다 작으면 C는 A의 최대 k개의 원소를 포함
    int j = k - i;
    int i_low = 0 > (k-n) ? 0 : k - n; // max(0, k-n) n이 k보다 작으면 C는 A의 최소 k-n개의 원소를 포함
    int j_low = 0 > (k-m) ? 0 : k - m; // max(0, k-m) m이 k보다 작으면 C는 B의 최소 k-m개의 원소를 포함
    int delta;
    bool active = true;
    //until B[j-1] < A[i] && A[i-1] < B[j]
    while(active){
        if(i > 0 && j < n && A[i-1] >= B[j]){
            delta = ((i - i_low + 1) >> 1); 
            j_low = j;
            i -= delta;
            j += delta;
        }
        else if (j > 0 && i < m && B[j-1] >= A[i]){
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            j -= delta;
            i += delta;
        }
        else {
            active = false;
        }
    }
    return i;
}

__global__
void merge_basic_kerner(int* A, int m, int* B, int n, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int elementsPerThread = ceil((m+n)/(blockDim.x*gridDim.x));
    int k_curr = tid*elementsPerThread; // start output index
    int k_next = min((tid+1)*elementsPerThread, m+n); // end output index
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr]);
}