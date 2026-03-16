#include <immintrin.h> 
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Vectorized Matrix-Vector Product (Optimized for M=1 case)
void matmul(float* A, float* B, float* C, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_val = A[i * K + k];
            __m256 a_vec = _mm256_set1_ps(a_val);
            float* Bk = B + k * N;
            float* Ci = C + i * N;
            for (int j = 0; j <= N - 8; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(Bk + j);
                __m256 c_vec = _mm256_loadu_ps(Ci + j);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                _mm256_storeu_ps(Ci + j, c_vec);
            }
            for (int j = (N / 8) * 8; j < N; j++) Ci[j] += a_val * Bk[j];
        }
    }
}

void gelu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.797884f * (v + 0.044715f * v * v * v)));
    }
}

void softmax(float* x, int size) {
    float max_v = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_v) max_v = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_v);
        sum += x[i];
    }
    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int i = 0; i < size; i++) x[i] *= inv_sum;
}

void update_output_layer(float* W2, float* h, float* d_logits, int dim, int vocab_size, float lr) {
    for (int i = 0; i < dim; i++) {
        float hi = h[i];
        if (fabsf(hi) < 1e-10f) continue;
        float* row = W2 + i * vocab_size;
        __m256 lr_h = _mm256_set1_ps(lr * hi);
        for (int j = 0; j <= vocab_size - 8; j += 8) {
            __m256 w = _mm256_loadu_ps(row + j);
            __m256 g = _mm256_loadu_ps(d_logits + j);
            _mm256_storeu_ps(row + j, _mm256_sub_ps(w, _mm256_mul_ps(lr_h, g)));
        }
        for (int j = (vocab_size / 8) * 8; j < vocab_size; j++) row[j] -= lr * hi * d_logits[j];
    }
}

void update_general(float* w, float* g, int size, float lr) {
    __m256 lr_v = _mm256_set1_ps(lr);
    for (int i = 0; i <= size - 8; i += 8) {
        __m256 wv = _mm256_loadu_ps(w + i);
        __m256 gv = _mm256_loadu_ps(g + i);
        _mm256_storeu_ps(w + i, _mm256_sub_ps(wv, _mm256_mul_ps(lr_v, gv)));
    }
}