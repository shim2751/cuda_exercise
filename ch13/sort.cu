__global__
void radit_basis_kernel(int* input, int* output, unsigned int* bits, unsigned int N, unsigned int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int key, bit;
    if (idx < N) {
        key = input[idx];
        bit = (key >> iter) & 1; // Extract the bit at position 'iter'
        bits[idx] = bit;   
    }

    exclusive_scan(bits, N); // outside of if state ensures that all threads are active
    if (idx < N) {
        unsigned int numOnesBefore = bits[idx];
        unsigned int numOnesTotal = bits[N-1];

        unsigned int dst = (bit == 0) ? (idx - numOnesBefore):(N - numOnesTotal) + numOnesBefore;
        output[dst] = key;
    }

}