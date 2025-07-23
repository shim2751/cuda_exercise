__global__
void histo_kernel(char* data, unsigned int length, unsigned int * histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (length > 0){
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26){
            atomicAdd(&(histo[pos/4]),1);
        }
    }
}

void histo_private_w_GM_kernel(char* data, unsigned int length, unsigned int * histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (length > 0){
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26){
            atomicAdd(&(histo[NUM_BIN * blockIdx.x + pos/4]),1);
        }
    }

    //block 0 already stored proper place.
    if(blockIdx.x > 0){
        __syncthreads();
        // each thread could work multiple times if blockDim.x is smaller than # of bin elements.
        for(unsigned int bin=threadIdx.x; bin < NUM_BIN; bin += blockDim.x){
            int binVal = histo[NUM_BIN * blockIdx.x + bin];
            if (binVal > 0 )
                atomicAdd(&(histo[bin]),binVal);
        }
    }
}

void histo_private_w_SM_kernel(char* data, unsigned int length, unsigned int * histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int histo_s[NUM_BIN];
    for(unsigned int bin; bin < NUM_BIN; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    if (length > 0){
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26){
            atomicAdd(&(histo_s[pos/4]),1);
        }
    }

    //block 0 already stored proper place.
    if(blockIdx.x > 0){
        __syncthreads();
        // each thread could work multiple times if blockDim.x is smaller than # of bin elements.
        for(unsigned int bin=threadIdx.x; bin < NUM_BIN; bin += blockDim.x){
            int binVal = histo_s[bin];
            if (binVal > 0 )
                atomicAdd(&(histo[bin]),binVal);
        }
    }
}

void histo_th_coar_contiguous_kernel(char* data, unsigned int length, unsigned int * histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int histo_s[NUM_BIN];
    for(unsigned int bin; bin < NUM_BIN; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for(unsigned int i=tid*CFACTOR; i<min((tid+1)*CFACTOR, length); ++i) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) {
            atomicAdd(&(histo_s[pos/4]), 1);
        }
    }

    __syncthreads();
    // each thread could work multiple times if blockDim.x is smaller than # of bin elements.
    for(unsigned int bin=threadIdx.x; bin < NUM_BIN; bin += blockDim.x){
        int binVal = histo_s[bin];
        if (binVal > 0 )
            atomicAdd(&(histo[bin]),binVal);
    }
}

void histo_th_coar_interleaved_kernel(char* data, unsigned int length, unsigned int * histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int histo_s[NUM_BIN];
    for(unsigned int bin; bin < NUM_BIN; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for(unsigned int i=tid*CFACTOR; i<length; i+=blockDim.x*gridDim.x) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26) {
            atomicAdd(&(histo_s[pos/4]), 1);
        }
    }

    __syncthreads();
    // each thread could work multiple times if blockDim.x is smaller than # of bin elements.
    for(unsigned int bin=threadIdx.x; bin < NUM_BIN; bin += blockDim.x){
        int binVal = histo_s[bin];
        if (binVal > 0 )
            atomicAdd(&(histo[bin]),binVal);
    }
}

void histo_aggre_kernel(char* data, unsigned int length, unsigned int * histo){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int histo_s[NUM_BIN];
    for(unsigned int bin; bin < NUM_BIN; bin += blockDim.x){
        histo_s[bin] = 0u;
    }
    __syncthreads();
    unsigned int acc_cnt = 0;
    int prev_pos = 0;
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    for(unsigned int i=tid*CFACTOR; i<length; i+=blockDim.x*gridDim.x) {
        int pos = data[i] - 'a';
        if (pos >= 0 && pos < 26){
            int bin = pos/4;
            if (prev_pos == pos)
                acc_cnt++;
            else{
                if (acc_cnt > 0){
                    atomicAdd(&(histo_s[prev_pos]), acc_cnt);
                }
                acc_cnt = 1;
                prev_pos = pos;
            }
        }
    }
    if (acc_cnt > 0){
        atomicAdd(&(histo_s[prev_pos]), acc_cnt);
    }

    __syncthreads();
    // each thread could work multiple times if blockDim.x is smaller than # of bin elements.
    for(unsigned int bin=threadIdx.x; bin < NUM_BIN; bin += blockDim.x){
        int binVal = histo_s[bin];
        if (binVal > 0 )
            atomicAdd(&(histo[bin]),binVal);
    }
}

