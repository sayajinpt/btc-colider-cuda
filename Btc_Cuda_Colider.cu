#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cassert>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <thread>
#include <mutex>
#include <atomic>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// -------------------- Configuration --------------------
struct Config {
    int intensity_level = 3;  // Default intensity (90%)
    int blocks = 0;           // Will be calculated based on intensity
    int tpb = 0;              // Will be calculated based on intensity
    uint32_t keys_per_thread = 0; // Will be calculated based on intensity
};

// -------------------- Big ints --------------------
struct Big256 { uint32_t w[8]; };
struct Big512 { uint32_t w[16]; };

// -------------------- GPU Configuration --------------------
struct GPUConfig {
    int device_id;
    int intensity_level;
    int blocks;
    int tpb;
    uint32_t keys_per_thread;
    Big256 start_key;
    Big256 end_key;
    uint64_t keys_processed;
    bool found;
    Big256 found_key;
    char found_address[36];
};

// -------------------- Global Variables --------------------
std::atomic<bool> global_found(false);
std::mutex output_mutex;

__device__ char d_target_address[36];

__device__ __host__ inline void big256_set_zero(Big256& a) { for (int i = 0; i < 8; i++) a.w[i] = 0; }
__device__ __host__ inline void copy256(const Big256& s, Big256& d) { for (int i = 0; i < 8; i++) d.w[i] = s.w[i]; }
__device__ __host__ inline int cmp256(const Big256& a, const Big256& b) {
    for (int i = 7; i >= 0; i--) { if (a.w[i] < b.w[i]) return -1; if (a.w[i] > b.w[i]) return 1; } return 0;
}
__device__ __host__ inline void add256(const Big256& a, const Big256& b, Big256& r) {
    uint64_t c = 0; for (int i = 0; i < 8; i++) { uint64_t t = (uint64_t)a.w[i] + b.w[i] + c; r.w[i] = (uint32_t)t; c = t >> 32; }
}
__device__ __host__ inline void sub256(const Big256& a, const Big256& b, Big256& r) {
    uint64_t br = 0; for (int i = 0; i < 8; i++) {
        uint64_t av = a.w[i]; uint64_t bv = b.w[i] + br;
        if (av >= bv) { r.w[i] = (uint32_t)(av - bv); br = 0; }
        else { r.w[i] = (uint32_t)((1ULL << 32) + av - bv); br = 1; }
    }
}
__device__ __host__ inline void add_uint64_to_big256(const Big256& base, uint64_t add, Big256& out) {
    uint64_t carry = add;
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)base.w[i] + (carry & 0xFFFFFFFFULL);
        out.w[i] = (uint32_t)sum;
        carry = (sum >> 32) + (carry >> 32);
        if (!carry) { for (int j = i + 1; j < 8; j++) out.w[j] = base.w[j]; return; }
    }
}

// -------------------- Field mod p (secp256k1) --------------------
__device__ __host__ inline void get_p(Big256& p) {
    const uint32_t pw[8] = {
        0xFFFFFC2Fu,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,
        0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu };
    for (int i = 0; i < 8; i++) p.w[i] = pw[i];
}
__device__ inline uint32_t dev_clz(uint32_t x) { if (x == 0) return 32; return __clz(x); }

__device__ inline void mul256(const Big256& a, const Big256& b, Big512& c) {
    for (int i = 0; i < 16; i++) c.w[i] = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t t = (uint64_t)a.w[i] * b.w[j] + c.w[i + j] + carry;
            c.w[i + j] = (uint32_t)t; carry = t >> 32;
        }
        c.w[i + 8] = (uint32_t)carry;
    }
}
__device__ inline void reduce_p(const Big512& prod, Big256& out) {
    auto add_word = [&](uint32_t& dst, uint64_t add, uint64_t& carry) { uint64_t s = (uint64_t)dst + add + carry; dst = (uint32_t)s; carry = s >> 32; };
    auto fold_once = [&](const Big512& in, Big512& o) {
        Big256 H; for (int i = 0; i < 8; i++) H.w[i] = in.w[8 + i];
        for (int i = 0; i < 16; i++) o.w[i] = 0;
        for (int i = 0; i < 8; i++) o.w[i] = in.w[i];
        uint64_t carry = 0;
        for (int i = 1; i < 16; i++) { uint64_t add = (i - 1 < 8) ? (uint64_t)H.w[i - 1] : 0; add_word(o.w[i], add, carry); }
        carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t add = (uint64_t)H.w[i] * 977ull;
            uint64_t s = (uint64_t)o.w[i] + (uint32_t)add + carry;
            o.w[i] = (uint32_t)s;
            carry = (s >> 32) + (add >> 32);
        }
        for (int i = 8; i < 16 && carry; i++) { uint64_t s = (uint64_t)o.w[i] + carry; o.w[i] = (uint32_t)s; carry = s >> 32; }
        };
    Big512 t1; fold_once(prod, t1);
    Big512 t2; fold_once(t1, t2);
    for (int i = 0; i < 8; i++) out.w[i] = t2.w[i];
    Big256 P; get_p(P);
    for (int k = 0; k < 2; k++) { if (cmp256(out, P) >= 0) { Big256 tmp; sub256(out, P, tmp); copy256(tmp, out); } else break; }
}
__device__ inline void modmul(const Big256& a, const Big256& b, Big256& r) { Big512 p; mul256(a, b, p); reduce_p(p, r); }
__device__ inline void modsq(const Big256& a, Big256& r) { modmul(a, a, r); }

__device__ __constant__ Big256 EXP_P_MINUS_2 = {
    {0xFFFFFC2Du,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu}
};
__device__ inline void modexp(const Big256& base, const Big256& exp, Big256& r) {
    Big256 acc; big256_set_zero(acc); acc.w[0] = 1;
    for (int wi = 7; wi >= 0; --wi) {
        uint32_t w = exp.w[wi];
        for (int b = 31; b >= 0; --b) {
            Big256 t; modsq(acc, t); copy256(t, acc);
            if ((w >> b) & 1u) { Big256 t2; modmul(acc, base, t2); copy256(t2, acc); }
        }
    }
    copy256(acc, r);
}
__device__ inline void modinv(const Big256& a, Big256& r) { modexp(a, EXP_P_MINUS_2, r); }

__device__ inline void addmod_p(const Big256& a, const Big256& b, Big256& r) {
    uint64_t c = 0; for (int i = 0; i < 8; i++) { uint64_t t = (uint64_t)a.w[i] + b.w[i] + c; r.w[i] = (uint32_t)t; c = t >> 32; }
    Big256 P; get_p(P);
    if (c || cmp256(r, P) >= 0) { Big256 t; sub256(r, P, t); copy256(t, r); }
}
__device__ inline void submod_p(const Big256& a, const Big256& b, Big256& r) {
    uint64_t br = 0; for (int i = 0; i < 8; i++) {
        uint64_t av = a.w[i]; uint64_t bv = b.w[i] + br;
        if (av >= bv) { r.w[i] = (uint32_t)(av - bv); br = 0; }
        else { r.w[i] = (uint32_t)((1ULL << 32) + av - bv); br = 1; }
    }
    if (br) { Big256 P; get_p(P); Big256 t; add256(r, P, t); copy256(t, r); }
}

// -------------------- EC: secp256k1 --------------------
struct PointJ { Big256 X, Y, Z; };

__device__ __constant__ Big256 Gx_const = {
    {0x16F81798u,0x59F2815Bu,0x2DCE28D9u,0x029BFCDBu,0xCE870B07u,0x55A06295u,0xF9DCBBACu,0x79BE667Eu}
};
__device__ __constant__ Big256 Gy_const = {
    {0xFB10D4B8u,0x9C47D08Fu,0xA6855419u,0xFD17B448u,0x0E1108A8u,0x5DA4FBFCu,0x26A3C465u,0x483ADA77u}
};
__device__ inline void getG(Big256& x, Big256& y) { copy256(Gx_const, x); copy256(Gy_const, y); }
__device__ inline bool is_inf(const PointJ& p) { for (int i = 0; i < 8; i++) if (p.Z.w[i] != 0) return false; return true; }
__device__ inline void to_jac(const Big256& x, const Big256& y, PointJ& o) { copy256(x, o.X); copy256(y, o.Y); big256_set_zero(o.Z); o.Z.w[0] = 1; }
__device__ inline void from_jac(const PointJ& p, Big256& ax, Big256& ay) {
    if (is_inf(p)) { big256_set_zero(ax); big256_set_zero(ay); return; }
    Big256 zinv; modinv(p.Z, zinv);
    Big256 z2; modsq(zinv, z2);
    Big256 z3; modmul(z2, zinv, z3);
    modmul(p.X, z2, ax); modmul(p.Y, z3, ay);
}
__device__ inline void jacobian_double(const PointJ& p, PointJ& r) {
    if (is_inf(p)) { r = p; return; }
    Big256 Y2, XY2, X2, M, S, S2, Y4, M2;
    modsq(p.Y, Y2); modmul(p.X, Y2, XY2); modsq(p.X, X2);
    Big256 t; addmod_p(X2, X2, t); addmod_p(t, X2, M);
    addmod_p(XY2, XY2, S); addmod_p(S, S, S);
    modsq(M, M2); modsq(Y2, Y4);
    Big256 nx; Big256 twoS; addmod_p(S, S, twoS); submod_p(M2, twoS, nx);
    Big256 ny; Big256 S_minus_nx; submod_p(S, nx, S_minus_nx);
    Big256 tmp; modmul(M, S_minus_nx, tmp);
    Big256 eightY4; addmod_p(Y4, Y4, eightY4); addmod_p(eightY4, eightY4, eightY4); addmod_p(eightY4, eightY4, eightY4);
    submod_p(tmp, eightY4, ny);
    Big256 nz; Big256 YZ; modmul(p.Y, p.Z, YZ); addmod_p(YZ, YZ, nz);
    r.X = nx; r.Y = ny; r.Z = nz;
}
__device__ inline void jacobian_add(const PointJ& p, const PointJ& q, PointJ& r) {
    if (is_inf(p)) { r = q; return; }
    if (is_inf(q)) { r = p; return; }
    Big256 Z2sq; modsq(q.Z, Z2sq);
    Big256 U1; modmul(p.X, Z2sq, U1);
    Big256 Z1sq; modsq(p.Z, Z1sq);
    Big256 U2; modmul(q.X, Z1sq, U2);
    Big256 Z2cu; modmul(Z2sq, q.Z, Z2cu);
    Big256 S1; modmul(p.Y, Z2cu, S1);
    Big256 Z1cu; modmul(Z1sq, p.Z, Z1cu);
    Big256 S2; modmul(q.Y, Z1cu, S2);
    if (cmp256(U1, U2) == 0) {
        if (cmp256(S1, S2) != 0) { PointJ inf; big256_set_zero(inf.X); big256_set_zero(inf.Y); big256_set_zero(inf.Z); r = inf; return; }
        else { jacobian_double(p, r); return; }
    }
    Big256 H; submod_p(U2, U1, H);
    Big256 R; submod_p(S2, S1, R);
    Big256 H2; modsq(H, H2);
    Big256 H3; modmul(H2, H, H3);
    Big256 U1H2; modmul(U1, H2, U1H2);
    Big256 R2; modsq(R, R2);
    Big256 t1; submod_p(R2, H3, t1);
    Big256 twoU1H2; addmod_p(U1H2, U1H2, twoU1H2);
    Big256 nx; submod_p(t1, twoU1H2, nx);
    Big256 U1H2_minus_nx; submod_p(U1H2, nx, U1H2_minus_nx);
    Big256 Rmul; modmul(R, U1H2_minus_nx, Rmul);
    Big256 S1H3; modmul(S1, H3, S1H3);
    Big256 ny; submod_p(Rmul, S1H3, ny);
    Big256 nz; modmul(H, p.Z, nz); modmul(nz, q.Z, nz);
    r.X = nx; r.Y = ny; r.Z = nz;
}
__device__ inline void scalar_mul(const PointJ& base, const Big256& k, PointJ& out) {
    PointJ R; big256_set_zero(R.X); big256_set_zero(R.Y); big256_set_zero(R.Z);
    PointJ A = base;
    for (int wi = 7; wi >= 0; --wi) {
        uint32_t w = k.w[wi];
        for (int b = 31; b >= 0; --b) {
            jacobian_double(R, R);
            if ((w >> b) & 1u) jacobian_add(R, A, R);
        }
    }
    out = R;
}

// -------------------- SHA256 / RIPEMD160 (CUDA) --------------------
__device__ __constant__ uint32_t K256_cuda[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
__device__ inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

__device__ inline void sha256_compress_cuda(uint32_t st[8], const uint8_t blk[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)blk[i * 4] << 24) | ((uint32_t)blk[i * 4 + 1] << 16) | ((uint32_t)blk[i * 4 + 2] << 8) | blk[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = st[0], b = st[1], c = st[2], d = st[3], e = st[4], f = st[5], g = st[6], h = st[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + K256_cuda[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    st[0] += a; st[1] += b; st[2] += c; st[3] += d; st[4] += e; st[5] += f; st[6] += g; st[7] += h;
}
__device__ void sha256_cuda(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint32_t st[8] = { 0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19 };
    uint8_t blk[64]; size_t off = 0;
    while (off + 64 <= len) { memcpy(blk, data + off, 64); sha256_compress_cuda(st, blk); off += 64; }
    size_t rem = len - off; memset(blk, 0, 64);
    if (rem) memcpy(blk, data + off, rem);
    blk[rem] = 0x80;
    if (rem >= 56) { sha256_compress_cuda(st, blk); memset(blk, 0, 64); }
    uint64_t bitlen = len * 8ULL; for (int i = 0; i < 8; i++) blk[63 - i] = (uint8_t)(bitlen >> (i * 8));
    sha256_compress_cuda(st, blk);
    for (int i = 0; i < 8; i++) { out[i * 4] = (uint8_t)(st[i] >> 24); out[i * 4 + 1] = (uint8_t)(st[i] >> 16); out[i * 4 + 2] = (uint8_t)(st[i] >> 8); out[i * 4 + 3] = (uint8_t)st[i]; }
}

// RIPEMD160
__device__ inline uint32_t ROL(uint32_t x, unsigned n) { return (x << n) | (x >> (32 - n)); }
__device__ void ripemd160_cuda(const uint8_t* msg, size_t size, uint8_t out[20]) {
    uint32_t h0 = 0x67452301UL, h1 = 0xEFCDAB89UL, h2 = 0x98BADCFEUL, h3 = 0x10325476UL, h4 = 0xC3D2E1F0UL;
    for (size_t offset = 0; offset <= size; offset += 64) {
        uint8_t block[64]; size_t rem = size - offset;
        if (rem >= 64) { memcpy(block, msg + offset, 64); }
        else {
            memset(block, 0, 64); if (rem > 0) memcpy(block, msg + offset, rem); block[rem] = 0x80;
            if (rem <= 55) { uint64_t bitlen = size * 8ULL; for (int i = 0; i < 8; i++) block[56 + i] = (uint8_t)(bitlen >> (i * 8)); }
        }
        uint32_t X[16]; for (int i = 0; i < 16; i++) {
            X[i] = (uint32_t)block[i * 4] | ((uint32_t)block[i * 4 + 1] << 8) | ((uint32_t)block[i * 4 + 2] << 16) | ((uint32_t)block[i * 4 + 3] << 24);
        }
        auto f = [&](int j, uint32_t x, uint32_t y, uint32_t z)->uint32_t {
            if (j <= 15) return x ^ y ^ z;
            if (j <= 31) return (x & y) | (~x & z);
            if (j <= 47) return (x | ~y) ^ z;
            if (j <= 63) return (x & z) | (y & ~z);
            return x ^ (y | ~z);
            };
        const unsigned r1[80] = {
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, 7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
            3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12, 1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
            4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
        };
        const unsigned r2[80] = {
            5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12, 6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
            15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13, 8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
            12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
        };
        const unsigned s1[80] = {
            11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8, 7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
            11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5, 11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
            9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
        };
        const unsigned s2[80] = {
            8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6, 9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
            9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5, 15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
            8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
        };
        const uint32_t K1[5] = { 0x00000000UL,0x5a827999UL,0x6ed9eba1UL,0x8f1bbcdcUL,0xa953fd4eUL };
        const uint32_t K2[5] = { 0x50a28be6UL,0x5c4dd124UL,0x6d703ef3UL,0x7a6d76e9UL,0x00000000UL };
        uint32_t A1 = h0, B1 = h1, C1 = h2, D1 = h3, E1 = h4;
        uint32_t A2 = h0, B2 = h1, C2 = h2, D2 = h3, E2 = h4;
        for (int j = 0; j < 80; j++) {
            uint32_t T = ROL(A1 + f(j, B1, C1, D1) + X[r1[j]] + K1[j / 16], s1[j]) + E1;
            A1 = E1; E1 = D1; D1 = ROL(C1, 10); C1 = B1; B1 = T;
            uint32_t TT = ROL(A2 + f(79 - j, B2, C2, D2) + X[r2[j]] + K2[j / 16], s2[j]) + E2;
            A2 = E2; E2 = D2; D2 = ROL(C2, 10); C2 = B2; B2 = TT;
        }
        uint32_t T = h1 + C1 + D2;
        h1 = h2 + D1 + E2;
        h2 = h3 + E1 + A2;
        h3 = h4 + A1 + B2;
        h4 = h0 + B1 + C2;
        h0 = T;
    }
    // LE output
    out[0] = (uint8_t)h0; out[1] = (uint8_t)(h0 >> 8); out[2] = (uint8_t)(h0 >> 16); out[3] = (uint8_t)(h0 >> 24);
    out[4] = (uint8_t)h1; out[5] = (uint8_t)(h1 >> 8); out[6] = (uint8_t)(h1 >> 16); out[7] = (uint8_t)(h1 >> 24);
    out[8] = (uint8_t)h2; out[9] = (uint8_t)(h2 >> 8); out[10] = (uint8_t)(h2 >> 16); out[11] = (uint8_t)(h2 >> 24);
    out[12] = (uint8_t)h3; out[13] = (uint8_t)(h3 >> 8); out[14] = (uint8_t)(h3 >> 16); out[15] = (uint8_t)(h3 >> 24);
    out[16] = (uint8_t)h4; out[17] = (uint8_t)(h4 >> 8); out[18] = (uint8_t)(h4 >> 16); out[19] = (uint8_t)(h4 >> 24);
}

// Base58 encoding
__device__ const char* B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
__device__ void base58_encode(const uint8_t* data, int size, char* out) {
    uint8_t tmp[25]; memcpy(tmp, data, size);
    int zeros = 0; while (zeros < size && tmp[zeros] == 0) zeros++;
    uint8_t buf[25 * 138 / 100 + 1] = { 0 }; int blen = 0;
    for (int i = zeros; i < size; i++) {
        uint32_t carry = tmp[i];
        for (int j = 0; j < blen; j++) { carry += (uint32_t)buf[j] * 256; buf[j] = (uint8_t)(carry % 58); carry /= 58; }
        while (carry) { buf[blen++] = (uint8_t)(carry % 58); carry /= 58; }
    }
    int pos = 0; for (int i = 0; i < zeros; i++) out[pos++] = '1';
    for (int i = blen - 1; i >= 0; i--) out[pos++] = B58[buf[i]];
    out[pos] = '\0';
}

// Target address to compare against
__device__ const char TARGET_ADDRESS[] = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU";

// -------------------- Address from Point (uncompressed) --------------------
__device__ bool point_to_address_uncompressed(const PointJ& P, char outB58[36], const char* target_address) {
    Big256 x, y; from_jac(P, x, y);

    // Serialize uncompressed 65 bytes
    uint8_t pub[65]; pub[0] = 0x04;
    for (int wi = 0; wi < 8; wi++) {
        uint32_t wx = x.w[7 - wi], wy = y.w[7 - wi];
        pub[1 + wi * 4] = (uint8_t)(wx >> 24); pub[2 + wi * 4] = (uint8_t)(wx >> 16);
        pub[3 + wi * 4] = (uint8_t)(wx >> 8);  pub[4 + wi * 4] = (uint8_t)wx;
        pub[33 + wi * 4] = (uint8_t)(wy >> 24); pub[34 + wi * 4] = (uint8_t)(wy >> 16);
        pub[35 + wi * 4] = (uint8_t)(wy >> 8);  pub[36 + wi * 4] = (uint8_t)wy;
    }

    uint8_t sha[32]; sha256_cuda(pub, 65, sha);
    uint8_t rh[20];  ripemd160_cuda(sha, 32, rh);

    uint8_t payload[21]; payload[0] = 0x00; memcpy(payload + 1, rh, 20);
    uint8_t chk1[32], chk2[32]; sha256_cuda(payload, 21, chk1); sha256_cuda(chk1, 32, chk2);

    uint8_t addrBytes[25]; memcpy(addrBytes, payload, 21); memcpy(addrBytes + 21, chk2, 4);

    // Encode to Base58
    base58_encode(addrBytes, 25, outB58);

    // Compare with target address
    for (int i = 0; target_address[i] != '\0'; i++) {
        if (outB58[i] != target_address[i]) {
            return false;
        }
    }
    return true;
}

// -------------------- Precompute & windowed multiply --------------------

// Build a small table T[d] = d * G for d = 0..table_size-1
// Single-threaded kernel (run once)
__global__ void build_precomp_table_kernel(PointJ* out_table, int table_size) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return; // only one thread builds it
    // Infinity
    PointJ inf; big256_set_zero(inf.X); big256_set_zero(inf.Y); big256_set_zero(inf.Z);
    out_table[0] = inf;
    Big256 gx, gy; getG(gx, gy);
    PointJ G; to_jac(gx, gy, G);
    if (table_size <= 1) return;
    out_table[1] = G;
    for (int d = 2; d < table_size; ++d) {
        // out_table[d] = out_table[d-1] + G
        jacobian_add(out_table[d - 1], G, out_table[d]);
    }
}

// Helper: extract w-bit window starting at bitpos (LSB bitpos = 0)
__device__ inline unsigned extract_window_from_big256(const Big256& k, int bitpos, int w) {
    // compute word index and bit offset
    int word_idx = bitpos / 32;
    int bit_off = bitpos % 32;
    uint64_t low = 0;
    if (word_idx < 8) low = k.w[word_idx];
    uint64_t high = 0;
    if (word_idx + 1 < 8) high = k.w[word_idx + 1];
    uint64_t val = (low >> bit_off) | (high << (32 - bit_off));
    unsigned mask = (1u << w) - 1u;
    return (unsigned)(val & mask);
}

// Windowed scalar multiply using precomputed table (table_size must equal 1<<window_bits)
__device__ inline void scalar_mul_windowed_precomp(const Big256& k, PointJ& out, const PointJ* table, int table_size, int window_bits) {
    const int bits = 256;
    const int windows = (bits + window_bits - 1) / window_bits;

    // R = infinity
    PointJ R; big256_set_zero(R.X); big256_set_zero(R.Y); big256_set_zero(R.Z);

    for (int wi = windows - 1; wi >= 0; --wi) {
        // R = 2^w * R  (do w doublings)
        for (int d = 0; d < window_bits; ++d) jacobian_double(R, R);

        int bitpos = wi * window_bits; // LSB-based position
        unsigned u = extract_window_from_big256(k, bitpos, window_bits);
        if (u != 0 && (int)u < table_size) {
            jacobian_add(R, table[u], R);
        }
    }
    out = R;
}

// -------------------- Collision detection kernel (uses precomputed table) --------------------
extern "C" __global__
void collision_kernel(Big256 startKey, Big256 endKey, uint32_t keysPerThread,
    unsigned long long* processed_out, bool* found, Big256* found_key, char* found_address,
    const PointJ* precomp_table, int precomp_table_size, int window_bits)
{
    const uint64_t gid = blockDim.x * (uint64_t)blockIdx.x + threadIdx.x;

    // Compute initial private key for this thread
    Big256 priv; add_uint64_to_big256(startKey, gid, priv);

    // Compute P0 = priv * G using windowed scalar multiply
    PointJ P; scalar_mul_windowed_precomp(priv, P, precomp_table, precomp_table_size, window_bits);

    // Preload G (Jacobian) for incremental adds
    Big256 Gx, Gy; getG(Gx, Gy);
    PointJ stepG; to_jac(Gx, Gy, stepG);

    // Walk KEYS_PER_THREAD (cap if we cross endKey)
    char address[36];
    uint32_t done = 0;
    for (uint32_t i = 0; i < keysPerThread; i++) {
        // stop if priv > end or collision already found
        if (cmp256(priv, endKey) > 0 || *found) break;

        // Compute address and check for collision
        if (point_to_address_uncompressed(P, address, d_target_address)) {
            // Collision found!
            *found = true;
            *found_key = priv;
            for (int j = 0; j < 36; j++) {
                found_address[j] = address[j];
            }
            break;
        }

        // Next key: P += G, priv += 1
        jacobian_add(P, stepG, P);
        Big256 one; big256_set_zero(one); one.w[0] = 1; Big256 npriv; add256(priv, one, npriv); priv = npriv;

        done++;
    }

    if (done) { atomicAdd(processed_out, (unsigned long long)done); }
}

// -------------------- Host helpers --------------------
static const char* DEFAULT_START =
"0000000000000000000000000000000000000000000000400000000000000000";
static const char* DEFAULT_END =
"00000000000000000000000000000000000000000000007fffffffffffffffff";

bool hex_to_big256(const char* hex, Big256& out) {
    const char* p = hex;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) p += 2;

    // Must be exactly 64 hex chars (32 bytes)
    if (strlen(p) != 64) {
        fprintf(stderr, "Hex string must be exactly 64 characters long\n");
        return false;
    }

    for (int i = 0; i < 8; i++) {
        uint32_t word = 0;
        for (int j = 0; j < 8; j++) {
            char c = p[(7 - i) * 8 + j];
            word <<= 4;
            if (c >= '0' && c <= '9') word |= c - '0';
            else if (c >= 'a' && c <= 'f') word |= 10 + c - 'a';
            else if (c >= 'A' && c <= 'F') word |= 10 + c - 'A';
            else {
                fprintf(stderr, "Invalid hex character: %c\n", c);
                return false;
            }
        }
        out.w[i] = word;
    }
    return true;
}

void print_hex256(const Big256& a) {
    // Print all 8 words (32 bytes) in big-endian order
    for (int wi = 7; wi >= 0; --wi) {
        printf("%08x", a.w[wi]);
    }
}

void bump_start_by_u64(Big256& start, uint64_t add) {
    Big256 tmp; add_uint64_to_big256(start, add, tmp); start = tmp;
}

// Helper function to format numbers with commas
std::string format_with_commas(uint64_t num) {
    std::string s = std::to_string(num);
    int n = s.length() - 3;
    while (n > 0) {
        s.insert(n, ",");
        n -= 3;
    }
    return s;
}

void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA devices found\n");
        return;
    }

    printf("\n=== CUDA Device Information ===\n");
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Global Memory: %.1f GB\n", (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Shared Memory per Block: %.1f KB\n", (float)prop.sharedMemPerBlock / 1024);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads per Multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Clock Rate: %.1f GHz\n", prop.clockRate * 1e-6f);
        printf("  Memory Clock Rate: %.1f GHz\n", prop.memoryClockRate * 1e-6f);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache Size: %.1f MB\n", (float)prop.l2CacheSize / (1024 * 1024));
        printf("  Total Constant Memory: %.1f KB\n", (float)prop.totalConstMem / 1024);
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  Device Overlap: %s\n", prop.deviceOverlap ? "Yes" : "No");
        printf("  Integrated GPU: %s\n", prop.integrated ? "Yes" : "No");
        printf("\n");
    }
}

// -------------------- Save File Helpers --------------------
const char* SAVE_FILE = "savefile.bin";
const char* RESULT_FILE = "collision_results.txt";

bool save_current_key(const Big256& key) {
    std::ofstream file(SAVE_FILE, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(&key), sizeof(Big256));
    return file.good();
}

bool load_saved_key(Big256& key) {
    std::ifstream file(SAVE_FILE, std::ios::binary);
    if (!file) return false;
    file.read(reinterpret_cast<char*>(&key), sizeof(Big256));
    return file.good();
}

void save_collision_result(const Big256& priv_key, const char* address, const char* target) {
    std::ofstream file(RESULT_FILE, std::ios::app);
    if (!file) return;

    // Get current time
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);

    file << "=== COLLISION FOUND ===\n";
    file << "Timestamp: " << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S") << "\n";
    file << "Target address: " << target << "\n";
    file << "Private key: 0x";
    for (int wi = 7; wi >= 0; --wi) {
        file << std::hex << std::setw(8) << std::setfill('0') << priv_key.w[wi];
    }
    file << "\nAddress: " << address << "\n\n";
}

// -------------------- Multi-GPU Helper Functions --------------------
uint64_t big256_to_uint64(const Big256& big) {
    uint64_t result = 0;
    for (int i = 0; i < 4; i++) {
        result += (uint64_t)big.w[i] << (i * 8);
    }
    return result;
}

void configure_intensity(GPUConfig& config, cudaDeviceProp& prop) {
    int max_threads_per_mp = prop.maxThreadsPerMultiProcessor;
    int mp_count = prop.multiProcessorCount;

    int target_blocks = mp_count * 6;
    int target_tpb = 256;
    uint32_t target_keys = 1024;

    switch (config.intensity_level) {
    case 1: // 30% workload
        target_blocks = (int)(mp_count * 2 * 0.4);
        target_tpb = (int)(256 * 0.4);
        target_keys = (uint32_t)(1024 * 0.4);
        break;
    case 2: // 60% workload
        target_blocks = (int)(mp_count * 4 * 0.8);
        target_tpb = (int)(256 * 0.8);
        target_keys = (uint32_t)(1024 * 0.8);
        break;
    case 3: // 90% workload (default)
        target_blocks = (int)(mp_count * 8 * 0.97);
        target_tpb = (int)(256 * 0.97);
        target_keys = (uint32_t)(1024 * 0.97);
        break;
    case 4: // 100% workload
        target_blocks = mp_count * 8;
        target_tpb = 512;
        target_keys = 2048;
        break;
    default:
        target_blocks = (int)(mp_count * 8 * 0.97);
        target_tpb = (int)(256 * 0.97);
        target_keys = (uint32_t)(1024 * 0.97);
        break;
    }

    config.blocks = std::max(1, target_blocks);
    config.tpb = std::max(32, target_tpb);
    config.keys_per_thread = std::max(1u, target_keys);

    if (config.intensity_level != 4) {
        config.tpb = std::min(config.tpb, prop.maxThreadsPerBlock);
        config.blocks = std::min(config.blocks, 65535);
    }
    else {
        config.blocks = std::min(config.blocks, 65535);
    }
}

std::vector<GPUConfig> initialize_gpu_configs(int device_count, const Big256& startKey,
    const Big256& endKey, int intensity_level) {
    std::vector<GPUConfig> configs(device_count);

    // Calculate approximate total range
    Big256 total_range;
    sub256(endKey, startKey, total_range);
    uint64_t approx_total = big256_to_uint64(total_range);
    uint64_t keys_per_gpu = approx_total / device_count;

    for (int i = 0; i < device_count; i++) {
        GPUConfig& config = configs[i];
        config.device_id = i;
        config.intensity_level = intensity_level;

        // Set key ranges
        if (i == 0) {
            config.start_key = startKey;
        }
        else {
            add_uint64_to_big256(configs[i - 1].end_key, 1, config.start_key);
        }

        if (i == device_count - 1) {
            config.end_key = endKey;
        }
        else {
            add_uint64_to_big256(config.start_key, keys_per_gpu, config.end_key);
        }

        // Configure device-specific intensity
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        configure_intensity(config, prop);

        config.found = false;
        config.keys_processed = 0;
    }

    return configs;
}

void gpu_worker(GPUConfig& config, const char* target_address,
    PointJ* d_precomp, int precomp_table_size, int window_bits) {

    cudaSetDevice(config.device_id);

    // Device variables
    unsigned long long* d_processed = nullptr;
    bool* d_found = nullptr;
    Big256* d_found_key = nullptr;
    char* d_found_address = nullptr;

    // Allocate device memory
    cudaMalloc(&d_processed, sizeof(unsigned long long));
    cudaMalloc(&d_found, sizeof(bool));
    cudaMalloc(&d_found_key, sizeof(Big256));
    cudaMalloc(&d_found_address, 36 * sizeof(char));

    // Copy target address to device
    char h_target_address[36] = { 0 };
    strncpy(h_target_address, target_address, 35);
    cudaMemcpyToSymbol(d_target_address, h_target_address, 36);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto last_log = std::chrono::steady_clock::now();
    uint64_t local_processed = 0;

    while (cmp256(config.start_key, config.end_key) <= 0 && !global_found.load()) {
        // Reset counters
        unsigned long long zero = 0;
        bool false_val = false;
        cudaMemcpy(d_processed, &zero, sizeof(zero), cudaMemcpyHostToDevice);
        cudaMemcpy(d_found, &false_val, sizeof(false_val), cudaMemcpyHostToDevice);

        // Launch kernel
        cudaEventRecord(start);
        collision_kernel << <config.blocks, config.tpb >> > (
            config.start_key, config.end_key, config.keys_per_thread,
            d_processed, d_found, d_found_key, d_found_address,
            d_precomp, precomp_table_size, window_bits
            );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Check for collision
        bool found = false;
        cudaMemcpy(&found, d_found, sizeof(found), cudaMemcpyDeviceToHost);

        if (found) {
            std::lock_guard<std::mutex> lock(output_mutex);
            global_found.store(true);
            config.found = true;

            cudaMemcpy(&config.found_key, d_found_key, sizeof(Big256), cudaMemcpyDeviceToHost);
            cudaMemcpy(config.found_address, d_found_address, 36, cudaMemcpyDeviceToHost);

            printf("\n=== COLLISION FOUND on GPU %d ===\n", config.device_id);
            printf("Private key: 0x");
            print_hex256(config.found_key);
            printf("\nAddress: %s\n", config.found_address);
            printf("Matches target: %s\n", target_address);

            save_collision_result(config.found_key, config.found_address, target_address);
            break;
        }

        // Update progress
        unsigned long long processed = 0;
        cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost);
        bump_start_by_u64(config.start_key, processed);
        config.keys_processed += processed;
        local_processed += processed;

        // Periodic logging
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count();

        if (elapsed >= 5) {
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);
            double secs = ms / 1000.0;
            double kps = (secs > 0.0) ? (processed / secs) : 0.0;

            std::lock_guard<std::mutex> lock(output_mutex);
            printf("GPU %d: %s keys/s (total: %s keys)\n",
                config.device_id,
                format_with_commas(static_cast<uint64_t>(kps)).c_str(),
                format_with_commas(config.keys_processed).c_str());

            last_log = now;
        }

        if (processed == 0) break;
    }

    // Cleanup
    cudaFree(d_processed);
    cudaFree(d_found);
    cudaFree(d_found_key);
    cudaFree(d_found_address);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// -------------------- Main --------------------
int main(int argc, char** argv) {
    // Default values
    const char* DEFAULT_ADDRESS = "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU";
    const char* DEFAULT_START = "0000000000000000000000000000000000000000000000400000000000000000";
    const char* DEFAULT_END = "00000000000000000000000000000000000000000000007fffffffffffffffff";

    Config config;
    config.intensity_level = 3;

    // Help flag
    if (argc == 2 && (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)) {
        printf("Usage: %s [options]\n", argv[0]);
        printf("Options:\n");
        printf("  -a, --address <addr>     Target Bitcoin address (default: %s)\n", DEFAULT_ADDRESS);
        printf("  -s, --start <hex>        Start key in hex (64 chars) (default: %s)\n", DEFAULT_START);
        printf("  -e, --end <hex>          End key in hex (64 chars) (default: %s)\n", DEFAULT_END);
        printf("  -i, --intensity <lvl>    GPU intensity level (1-4, default: 3)\n");
        printf("  -h, --help               Show this help message\n");
        return 0;
    }

    // Initialize with defaults
    const char* targetAddress = DEFAULT_ADDRESS;
    const char* startArg = DEFAULT_START;
    const char* endArg = DEFAULT_END;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--address") == 0) {
            if (i + 1 < argc) targetAddress = argv[++i];
        }
        else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--start") == 0) {
            if (i + 1 < argc) startArg = argv[++i];
        }
        else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--end") == 0) {
            if (i + 1 < argc) endArg = argv[++i];
        }
        else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--intensity") == 0) {
            if (i + 1 < argc) config.intensity_level = atoi(argv[++i]);
        }
    }

    // Convert hex strings to Big256
    Big256 startKey, endKey;
    if (!hex_to_big256(startArg, startKey) || !hex_to_big256(endArg, endKey)) {
        fprintf(stderr, "Invalid hex input for key range.\n");
        return 1;
    }

    // Check for save file
    Big256 loadedKey;
    bool hasSaveFile = load_saved_key(loadedKey);
    if (hasSaveFile) {
        char choice;
        printf("Found save file. Do you want to resume from saved position? (y/n): ");
        std::cin >> choice;
        if (choice == 'y' || choice == 'Y') {
            if (cmp256(loadedKey, startKey) >= 0 && cmp256(loadedKey, endKey) <= 0) {
                startKey = loadedKey;
                printf("Resuming from saved key: 0x");
                print_hex256(startKey);
                printf("\n");
            }
        }
    }

    // Validate address
    if (strlen(targetAddress) < 26 || strlen(targetAddress) > 35) {
        fprintf(stderr, "Error: Invalid address length.\n");
        return 1;
    }

    if (cmp256(startKey, endKey) > 0) {
        fprintf(stderr, "Error: start key > end key.\n");
        return 1;
    }

    // Get number of available GPUs
    int device_count;
    cudaGetDeviceCount(&device_count);
    device_count = std::min(device_count, 8); // Limit to 8 GPUs

    printf("Found %d CUDA device(s)\n", device_count);
    print_device_info();

    // Precompute parameters
    const int WINDOW_BITS = 4;
    const int precomp_table_size = (1 << WINDOW_BITS);

    if (device_count > 1) {
        printf("Using multi-GPU mode with %d devices\n", device_count);

        // Initialize GPU configurations
        std::vector<GPUConfig> gpu_configs = initialize_gpu_configs(device_count, startKey, endKey, config.intensity_level);

        // Create precomputation tables for each GPU
        std::vector<PointJ*> d_precomp_tables(device_count);
        for (int i = 0; i < device_count; i++) {
            cudaSetDevice(i);
            cudaMalloc(&d_precomp_tables[i], sizeof(PointJ) * precomp_table_size);
            build_precomp_table_kernel << <1, 1 >> > (d_precomp_tables[i], precomp_table_size);
            cudaDeviceSynchronize();
        }

        // Launch worker threads for each GPU
        std::vector<std::thread> workers;
        auto program_start = std::chrono::steady_clock::now();

        for (int i = 0; i < device_count; i++) {
            workers.emplace_back([&, i]() {
                gpu_worker(gpu_configs[i], targetAddress, d_precomp_tables[i],
                    precomp_table_size, WINDOW_BITS);
                });
        }

        // Wait for all workers to complete
        for (auto& worker : workers) {
            worker.join();
        }

        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - program_start).count();

        // Calculate total keys processed
        uint64_t total_done = 0;
        for (const auto& config : gpu_configs) {
            total_done += config.keys_processed;
        }

        printf("\n=== Multi-GPU Search Completed ===\n");
        printf("Total keys searched: %s\n", format_with_commas(total_done).c_str());
        printf("Total time: %s seconds\n", format_with_commas(total_elapsed).c_str());
        printf("Average speed: %s keys/s\n",
            format_with_commas(static_cast<uint64_t>((total_elapsed > 0) ? total_done / total_elapsed : 0)).c_str());

        // Cleanup
        for (int i = 0; i < device_count; i++) {
            cudaSetDevice(i);
            cudaFree(d_precomp_tables[i]);
        }

    }
    else {
        // Single GPU fallback
        printf("Using single GPU mode\n");

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        // Configure intensity for single GPU
        config.blocks = 0;
        config.tpb = 0;
        config.keys_per_thread = 0;

        int max_threads_per_mp = prop.maxThreadsPerMultiProcessor;
        int mp_count = prop.multiProcessorCount;

        switch (config.intensity_level) {
        case 1: // 30% workload
            config.blocks = (int)(mp_count * 2 * 0.4);
            config.tpb = (int)(256 * 0.4);
            config.keys_per_thread = (uint32_t)(1024 * 0.4);
            break;
        case 2: // 60% workload
            config.blocks = (int)(mp_count * 4 * 0.8);
            config.tpb = (int)(256 * 0.8);
            config.keys_per_thread = (uint32_t)(1024 * 0.8);
            break;
        case 3: // 90% workload (default)
            config.blocks = (int)(mp_count * 8 * 0.97);
            config.tpb = (int)(256 * 0.97);
            config.keys_per_thread = (uint32_t)(1024 * 0.97);
            break;
        case 4: // 100% workload
            config.blocks = mp_count * 8;
            config.tpb = 512;
            config.keys_per_thread = 2048;
            break;
        default:
            config.blocks = (int)(mp_count * 8 * 0.97);
            config.tpb = (int)(256 * 0.97);
            config.keys_per_thread = (uint32_t)(1024 * 0.97);
            break;
        }

        config.blocks = std::max(1, config.blocks);
        config.tpb = std::max(32, config.tpb);
        config.keys_per_thread = std::max(1u, config.keys_per_thread);

        if (config.intensity_level != 4) {
            config.tpb = std::min(config.tpb, prop.maxThreadsPerBlock);
            config.blocks = std::min(config.blocks, 65535);
        }
        else {
            config.blocks = std::min(config.blocks, 65535);
        }

        // Print search parameters
        printf("\n=== Collision Search Parameters ===\n");
        printf("Target address: %s\n", targetAddress);
        printf("Start key:      0x");
        print_hex256(startKey);
        printf("\nEnd key:        0x");
        print_hex256(endKey);
        printf("\nIntensity level: %d\n", config.intensity_level);

        // Calculate batch size
        const uint64_t batch_keys = (uint64_t)config.blocks * config.tpb * config.keys_per_thread;

        printf("\n=== GPU Configuration ===\n");
        printf("Blocks:          %d\n", config.blocks);
        printf("Threads/block:   %d\n", config.tpb);
        printf("Keys/thread:     %u\n", config.keys_per_thread);
        printf("Total keys/batch: %s\n", format_with_commas(batch_keys).c_str());
        fflush(stdout);

        // Device variables
        unsigned long long* d_processed = nullptr;
        bool* d_found = nullptr;
        Big256* d_found_key = nullptr;
        char* d_found_address = nullptr;
        PointJ* d_precomp = nullptr;

        // Initialize CUDA
        cudaError_t cudaStatus;

        // Copy target address to device
        char h_target_address[36] = { 0 };
        strncpy(h_target_address, targetAddress, 35);
        cudaStatus = cudaMemcpyToSymbol(d_target_address, h_target_address, 36);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpyToSymbol failed for target address: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Allocate device memory
        cudaStatus = cudaMalloc(&d_processed, sizeof(unsigned long long));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for d_processed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaMalloc(&d_found, sizeof(bool));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for d_found: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaMalloc(&d_found_key, sizeof(Big256));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for d_found_key: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaMalloc(&d_found_address, 36 * sizeof(char));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for d_found_address: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Allocate device precomp table
        cudaStatus = cudaMalloc(&d_precomp, sizeof(PointJ) * precomp_table_size);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for d_precomp: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Build precomp table on device
        build_precomp_table_kernel << <1, 1 >> > (d_precomp, precomp_table_size);
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "build_precomp_table_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize failed after precomp build: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Initialize device variables
        unsigned long long zero = 0;
        bool false_val = false;
        cudaStatus = cudaMemcpy(d_processed, &zero, sizeof(zero), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed for d_processed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaMemcpy(d_found, &false_val, sizeof(false_val), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed for d_found: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // Timing
        cudaEvent_t start, stop;
        cudaStatus = cudaEventCreate(&start);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventCreate failed for start: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        cudaStatus = cudaEventCreate(&stop);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaEventCreate failed for stop: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        auto program_start = std::chrono::steady_clock::now();
        auto last_log = program_start;
        uint64_t total_done = 0;
        uint64_t total_since_last = 0;
        auto last_save = std::chrono::steady_clock::now();

        printf("\n=== Starting Search ===\n");
        while (cmp256(startKey, endKey) <= 0 && !global_found.load()) {
            // Reset counters
            cudaStatus = cudaMemcpy(d_processed, &zero, sizeof(zero), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed for d_processed reset: %s\n", cudaGetErrorString(cudaStatus));
                break;
            }

            cudaStatus = cudaMemcpy(d_found, &false_val, sizeof(false_val), cudaMemcpyHostToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed for d_found reset: %s\n", cudaGetErrorString(cudaStatus));
                break;
            }

            // Run kernel
            cudaStatus = cudaEventRecord(start);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaEventRecord failed for start: %s\n", cudaGetErrorString(cudaStatus));
                break;
            }

            collision_kernel << <config.blocks, config.tpb >> > (startKey, endKey, config.keys_per_thread,
                d_processed, d_found, d_found_key, d_found_address,
                d_precomp, precomp_table_size, WINDOW_BITS);

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "collision_kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                break;
            }

            cudaStatus = cudaEventRecord(stop);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaEventRecord failed for stop: %s\n", cudaGetErrorString(cudaStatus));
                break;
            }

            cudaStatus = cudaEventSynchronize(stop);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaEventSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
                break;
            }

            // Check for collision
            bool found = false;
            cudaStatus = cudaMemcpy(&found, d_found, sizeof(found), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed for d_found read: %s\n", cudaGetErrorString(cudaStatus));
                break;
            }

            if (found) {
                Big256 found_key;
                char found_address[36] = { 0 };
                cudaStatus = cudaMemcpy(&found_key, d_found_key, sizeof(found_key), cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed for found_key: %s\n", cudaGetErrorString(cudaStatus));
                    break;
                }

                cudaStatus = cudaMemcpy(found_address, d_found_address, 36, cudaMemcpyDeviceToHost);
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaMemcpy failed for found_address: %s\n", cudaGetErrorString(cudaStatus));
                    break;
                }

                printf("\n=== COLLISION FOUND ===\n");
                printf("Private key: 0x");
                print_hex256(found_key);
                printf("\nAddress: %s\n", found_address);
                printf("Matches target: %s\n", targetAddress);

                save_collision_result(found_key, found_address, targetAddress);
                global_found.store(true);
                break;
            }

            // Get processed count
            unsigned long long processed = 0;
            cudaStatus = cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaMemcpy failed for d_processed read: %s\n", cudaGetErrorString(cudaStatus));
                break;
            }

            // Update position
            bump_start_by_u64(startKey, processed);
            total_done += processed;
            total_since_last += processed;

            // Performance metrics
            float ms = 0;
            cudaStatus = cudaEventElapsedTime(&ms, start, stop);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaEventElapsedTime failed: %s\n", cudaGetErrorString(cudaStatus));
                break;
            }

            double secs = ms / 1000.0;
            double kps = (secs > 0.0) ? (processed / secs) : 0.0;

            // Log progress
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count();

            // Save progress every 30 seconds
            auto time_since_save = std::chrono::duration_cast<std::chrono::seconds>(now - last_save).count();
            if (time_since_save >= 30) {
                if (save_current_key(startKey)) {
                    printf("Progress saved at key: 0x");
                    print_hex256(startKey);
                    printf("\n");
                }
                last_save = now;
            }

            if (elapsed >= 10) {
                double total_elapsed_time = std::chrono::duration<double>(now - program_start).count();
                double avg_kps = (total_elapsed_time > 0.0) ? (total_done / total_elapsed_time) : 0.0;
                double recent_kps = (elapsed > 0) ? (total_since_last / elapsed) : 0.0;

                printf("[%.0fs] Total: %s keys | Avg: %s keys/s | Recent: %s keys/s\n",
                    total_elapsed_time,
                    format_with_commas(total_done).c_str(),
                    format_with_commas(static_cast<uint64_t>(avg_kps)).c_str(),
                    format_with_commas(static_cast<uint64_t>(recent_kps)).c_str());

                fflush(stdout);

                last_log = now;
                total_since_last = 0;
            }

            if (processed == 0) break;
        }

        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - program_start).count();
        printf("\n=== Search Completed ===\n");
        printf("Total keys searched: %s\n", format_with_commas(total_done).c_str());
        printf("Total time: %s seconds\n", format_with_commas(total_elapsed).c_str());
        printf("Average speed: %s keys/s\n",
            format_with_commas(static_cast<uint64_t>((total_elapsed > 0) ? total_done / (double)total_elapsed : 0)).c_str());

    Error:
        // Cleanup
        if (d_processed) cudaFree(d_processed);
        if (d_found) cudaFree(d_found);
        if (d_found_key) cudaFree(d_found_key);
        if (d_found_address) cudaFree(d_found_address);
        if (d_precomp) cudaFree(d_precomp);
    }

    cudaDeviceReset();
    return 0;
}
