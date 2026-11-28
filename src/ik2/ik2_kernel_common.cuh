#pragma once

#include "ik2_kernels.cuh"

#include <cuda_fp16.h>
#ifndef STRATA_USE_HIP
#include <cuda_fp8.h>
#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)
#include <cuda_fp4.h>
#endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace ik2::k {
namespace {

#ifdef STRATA_USE_HIP
[[noreturn, maybe_unused]] inline void unavailable_mma(const char* name) {
    std::fprintf(stderr, "ik2::k::%s requires CUDA tensor cores\n", name);
    std::abort();
}
#endif

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ik2::k::%s: launch failed: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }
__device__ __forceinline__ float2 u32_to_f2(uint32_t u) {
    const __half2 h2 = __halves2half2(__ushort_as_half((uint16_t) (u & 0xFFFFu)),
                                      __ushort_as_half((uint16_t) (u >> 16)));
    return __half22float2(h2);
}
__host__ __device__ __forceinline__ float bits_to_float(uint32_t u) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#endif
}
[[maybe_unused]] __host__ __device__ __forceinline__ uint32_t float_to_bits(float f) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __float_as_uint(f);
#else
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
#endif
}
__host__ __device__ inline float half_bits_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = sign;
        } else {   // subnormal
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 1024)) mm <<= 1, ++sh;
            u = sign | ((uint32_t) (113 - sh) << 23) | ((mm & 1023) << 13);
        }
    } else if (e == 31) {
        u = sign | 0x7f800000u | (m << 13);
    } else {
        u = sign | ((e + 112) << 23) | (m << 13);
    }
    return bits_to_float(u);
}
__host__ __device__ __forceinline__ uint8_t f32_to_e4m3fn(float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return (uint8_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#else
    const uint32_t u = float_to_bits(v);
    const uint8_t sign = (uint8_t) ((u >> 24) & 0x80u);
    const uint32_t ax = u & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) return sign | 0x7Fu;
    if (ax >= 0x43E00000u) return sign | 0x7Eu;
    if (ax < 0x3C800000u) {
#if defined(__CUDA_ARCH__)
        const uint32_t sub = float_to_bits(__fadd_rn(bits_to_float(ax), 16384.0f)) - 0x46800000u;
#else
        const uint32_t sub = float_to_bits(bits_to_float(ax) + 16384.0f) - 0x46800000u;
#endif
        return sign | (uint8_t) sub;
    }
    const uint32_t lsb = (ax >> 20) & 1u;
    return sign | (uint8_t) (((ax + 0x7FFFFu + lsb) >> 20) - (120u << 3));
#endif
}
__device__ __forceinline__ uint16_t f32x2_to_e4m3fnx2(float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return (uint16_t) __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
#else
    return (uint16_t) f32_to_e4m3fn(a) | ((uint16_t) f32_to_e4m3fn(b) << 8);
#endif
}
__host__ __device__ __forceinline__ uint16_t e4m3fn_to_f16(uint8_t q) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return __half_as_ushort(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) q, __NV_E4M3)));
#else
    const uint16_t sign = (uint16_t) (q & 0x80u) << 8;
    const uint32_t mag = q & 0x7Fu;
    if (mag == 0x7Fu) return sign | 0x7FFFu;
    if (mag < 8u) {
#if defined(__CUDA_ARCH__)
        return sign | (uint16_t) (__byte_perm(0x1E1C1800u, 0x23222120u, mag << 4) & 0xFF00u);
#else
#if defined(__HIP_DEVICE_COMPILE__)
        return sign | (uint16_t)((__builtin_amdgcn_perm(0x23222120u, 0x1e1c1800u, mag) & 0xffu) << 8);
#else
        static constexpr uint16_t kSub[8] = {0x0000u, 0x1800u, 0x1C00u, 0x1E00u, 0x2000u, 0x2100u, 0x2200u, 0x2300u};
        return sign | kSub[mag];
#endif
#endif
    }
    return sign | (uint16_t) ((mag << 7) + 0x2000u);
#endif
}
__device__ __forceinline__ uint32_t e4m3fnx2_to_h2_u32(uint16_t p2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const __half2 h2 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) p2, __NV_E4M3));
    return (uint32_t) __half_as_ushort(__low2half(h2)) | ((uint32_t) __half_as_ushort(__high2half(h2)) << 16);
#elif defined(__HIP_DEVICE_COMPILE__)
    const uint32_t bytes = __builtin_amdgcn_perm(0, p2, 0x04010400u);
    const uint32_t raw = ((bytes & 0x007f007fu) << 7) | ((bytes & 0x00800080u) << 8);
    // E4M3's exponent bias is eight below FP16's. Scaling normalizes its subnormals too.
    using Half2 = _Float16 __attribute__((ext_vector_type(2)));
    const Half2 scaled = __builtin_bit_cast(Half2, raw) * (_Float16)256;
    const uint32_t nan_lo = (p2 & 0x7fu) == 0x7fu ? 0x00007fffu : 0;
    const uint32_t nan_hi = (p2 & 0x7f00u) == 0x7f00u ? 0x7fff0000u : 0;
    return __builtin_bit_cast(uint32_t, scaled) | nan_lo | nan_hi;
#else
    return (uint32_t) e4m3fn_to_f16((uint8_t) (p2 & 0xFFu)) |
           ((uint32_t) e4m3fn_to_f16((uint8_t) (p2 >> 8)) << 16);
#endif
}
__host__ __device__ __forceinline__ float e4m3fn_to_f32(uint8_t q) {
#if defined(__CUDA_ARCH__)
    return h2f(e4m3fn_to_f16(q));
#else
    return half_bits_to_float(e4m3fn_to_f16(q));
#endif
}
__device__ __forceinline__ float2 e4m3fnx2_to_f2(uint16_t p2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) p2, __NV_E4M3)));
#else
    return u32_to_f2(e4m3fnx2_to_h2_u32(p2));
#endif
}

// the block's sum / max of v (blockDim.x a power of two, sh [blockDim.x]); every thread gets it
__device__ float bsum(float v, float* sh) {
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (t < o) sh[t] += sh[t + o];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float bmax(float v, float* sh) {
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (t < o) sh[t] = fmaxf(sh[t], sh[t + o]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

constexpr int kBlk = 256;

__device__ __forceinline__ uint32_t f2_to_u32(float a, float b) {
    const __half2 h2 = __float22half2_rn(make_float2(a, b));
    return (uint32_t) __half_as_ushort(__low2half(h2)) | ((uint32_t) __half_as_ushort(__high2half(h2)) << 16);
}

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ uint16_t f2bf_rn(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;   // NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

unsigned grid(size_t n, int b = kBlk) { return (unsigned) ((n + b - 1) / b); }

}  // namespace
}  // namespace ik2::k
