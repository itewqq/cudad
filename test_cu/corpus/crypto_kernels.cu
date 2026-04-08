// Complex CUDA kernels — crypto: AES-128 single-block encrypt, SHA-256 round
// These produce deeply nested loops, lookup tables (shared mem), and
// bit-manipulation that stress the structurizer + lifter.

#include <stdint.h>

// ------ AES-128 helpers (T-table approach, one block) ------

__device__ static const uint32_t Te0[256] = {
    0xc66363a5,0xf87c7c84,0xee777799,0xf67b7b8d,
    0xfff2f20d,0xd66b6bbd,0xde6f6fb1,0x91c5c554,
    0x60303050,0x02010103,0xce6767a9,0x562b2b7d,
    0xe7fefe19,0xb5d7d762,0x4dababe6,0xec76769a,
    0x8fcaca45,0x1f82829d,0x89c9c940,0xfa7d7d87,
    0xeffafa15,0xb25959eb,0x8e4747c9,0xfbf0f00b,
    0x41adadec,0xb3d4d467,0x5fa2a2fd,0x45afafea,
    0x239c9cbf,0x53a4a4f7,0xe4727296,0x9bc0c05b,
    0x75b7b7c2,0xe1fdfd1c,0x3d9393ae,0x4c26266a,
    0x6c36365a,0x7e3f3f41,0xf5f7f702,0x83cccc4f,
    0x6834345c,0x51a5a5f4,0xd1e5e534,0xf9f1f108,
    0xe2717193,0xabd8d873,0x62313153,0x2a15153f,
    0x0804040c,0x95c7c752,0x46232365,0x9dc3c35e,
    0x30181828,0x379696a1,0x0a05050f,0x2f9a9ab5,
    0x0e070709,0x24121236,0x1b80809b,0xdfe2e23d,
    0xcdebeb26,0x4e272769,0x7fb2b2cd,0xea75759f,
    0x1209091b,0x1d83839e,0x582c2c74,0x341a1a2e,
    0x361b1b2d,0xdc6e6eb2,0xb45a5aee,0x5ba0a0fb,
    0xa45252f6,0x763b3b4d,0xb7d6d661,0x7db3b3ce,
    0x5229297b,0xdde3e33e,0x5e2f2f71,0x13848497,
    0xa65353f5,0xb9d1d168,0x00000000,0xc1eded2c,
    0x40202060,0xe3fcfc1f,0x79b1b1c8,0xb65b5bed,
    0xd46a6abe,0x8dcbcb46,0x67bebed9,0x7239394b,
    0x944a4ade,0x984c4cd4,0xb05858e8,0x85cfcf4a,
    0xbbd0d06b,0xc5efef2a,0x4faaaae5,0xedfbfb16,
    0x864343c5,0x9a4d4dd7,0x66333355,0x11858594,
    0x8a4545cf,0xe9f9f910,0x04020206,0xfe7e7e81,
    0xa05050f0,0x783c3c44,0x259f9fba,0x4ba8a8e3,
    0xa25151f3,0x5da3a3fe,0x804040c0,0x058f8f8a,
    0x3f9292ad,0x219d9dbc,0x70383848,0xf1f5f504,
    0x63bcbcdf,0x77b6b6c1,0xafdada75,0x42212163,
    0x20101030,0xe5ffff1a,0xfdf3f30e,0xbfd2d26d,
    0x81cdcd4c,0x180c0c14,0x26131335,0xc3ecec2f,
    0xbe5f5fe1,0x359797a2,0x884444cc,0x2e171739,
    0x93c4c457,0x55a7a7f2,0xfc7e7e82,0x7a3d3d47,
    0xc86464ac,0xba5d5de7,0x3219192b,0xe6737395,
    0xc06060a0,0x19818198,0x9e4f4fd1,0xa3dcdc7f,
    0x44222266,0x542a2a7e,0x3b9090ab,0x0b888883,
    0x8c4646ca,0xc7eeee29,0x6bb8b8d3,0x2814143c,
    0xa7dede79,0xbc5e5ee2,0x160b0b1d,0xaddbdb76,
    0xdbe0e03b,0x64323256,0x743a3a4e,0x140a0a1e,
    0x924949db,0x0c06060a,0x4824246c,0xb85c5ce4,
    0x9fc2c25d,0xbdd3d36e,0x43acacef,0xc46262a6,
    0x399191a8,0x319595a4,0xd3e4e437,0xf279798b,
    0xd5e7e732,0x8bc8c843,0x6e373759,0xda6d6db7,
    0x018d8d8c,0xb1d5d564,0x9c4e4ed2,0x49a9a9e0,
    0xd86c6cb4,0xac5656fa,0xf3f4f407,0xcfeaea25,
    0xca6565af,0xf47a7a8e,0x47aeaee9,0x10080818,
    0x6fbabad5,0xf0787888,0x4a25256f,0x5c2e2e72,
    0x381c1c24,0x57a6a6f1,0x73b4b4c7,0x97c6c651,
    0xcbe8e823,0xa1dddd7c,0xe874749c,0x3e1f1f21,
    0x964b4bdd,0x61bdbddc,0x0d8b8b86,0x0f8a8a85,
    0xe0707090,0x7c3e3e42,0x71b5b5c4,0xcc6666aa,
    0x904848d8,0x06030305,0xf7f6f601,0x1c0e0e12,
    0xc26161a3,0x6a35355f,0xae5757f9,0x69b9b9d0,
    0x17868691,0x99c1c158,0x3a1d1d27,0x279e9eb9,
    0xd9e1e138,0xebf8f813,0x2b9898b3,0x22111133,
    0xd26969bb,0xa9d9d970,0x078e8e89,0x339494a7,
    0x2d9b9bb6,0x3c1e1e22,0x15878792,0xc9e9e920,
    0x87cece49,0xaa5555ff,0x50282878,0xa5dfdf7a,
    0x038c8c8f,0x59a1a1f8,0x09898980,0x1a0d0d17,
    0x65bfbfda,0xd7e6e631,0x844242c6,0xd06868b8,
    0x824141c3,0x299999b0,0x5a2d2d77,0x1e0f0f11,
    0x7bb0b0cb,0xa85454fc,0x6dbbbbd6,0x2c16163a,
};

// Simplified RCON
__device__ static const uint8_t rcon[10] = {
    0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36
};

extern "C" __global__ void aes128_encrypt_block(
    const uint32_t * __restrict__ plaintext,  // 4 uint32
    const uint32_t * __restrict__ round_keys, // 44 uint32 (11 round keys)
    uint32_t       * __restrict__ ciphertext  // 4 uint32
) {
    // Each thread encrypts one block (toy perf, real patterns).
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return; // single-block demo

    uint32_t s0 = plaintext[0] ^ round_keys[0];
    uint32_t s1 = plaintext[1] ^ round_keys[1];
    uint32_t s2 = plaintext[2] ^ round_keys[2];
    uint32_t s3 = plaintext[3] ^ round_keys[3];

    // 9 main rounds
    for (int r = 1; r < 10; r++) {
        uint32_t t0 = Te0[(s0>>24)&0xff] ^ __byte_perm(Te0[(s1>>16)&0xff],0,0x0321)
                     ^ __byte_perm(Te0[(s2>>8)&0xff],0,0x1032) ^ __byte_perm(Te0[s3&0xff],0,0x2103);
        uint32_t t1 = Te0[(s1>>24)&0xff] ^ __byte_perm(Te0[(s2>>16)&0xff],0,0x0321)
                     ^ __byte_perm(Te0[(s3>>8)&0xff],0,0x1032) ^ __byte_perm(Te0[s0&0xff],0,0x2103);
        uint32_t t2 = Te0[(s2>>24)&0xff] ^ __byte_perm(Te0[(s3>>16)&0xff],0,0x0321)
                     ^ __byte_perm(Te0[(s0>>8)&0xff],0,0x1032) ^ __byte_perm(Te0[s1&0xff],0,0x2103);
        uint32_t t3 = Te0[(s3>>24)&0xff] ^ __byte_perm(Te0[(s0>>16)&0xff],0,0x0321)
                     ^ __byte_perm(Te0[(s1>>8)&0xff],0,0x1032) ^ __byte_perm(Te0[s2&0xff],0,0x2103);
        s0 = t0 ^ round_keys[r*4+0];
        s1 = t1 ^ round_keys[r*4+1];
        s2 = t2 ^ round_keys[r*4+2];
        s3 = t3 ^ round_keys[r*4+3];
    }

    // Final round (no MixColumns)
    uint32_t rk = 40;
    uint32_t f0 = (Te0[(s0>>24)&0xff]&0xff000000) ^ (Te0[(s1>>16)&0xff]&0x00ff0000)
                ^ (Te0[(s2>>8)&0xff]&0x0000ff00) ^ (Te0[s3&0xff]&0x000000ff) ^ round_keys[rk];
    uint32_t f1 = (Te0[(s1>>24)&0xff]&0xff000000) ^ (Te0[(s2>>16)&0xff]&0x00ff0000)
                ^ (Te0[(s3>>8)&0xff]&0x0000ff00) ^ (Te0[s0&0xff]&0x000000ff) ^ round_keys[rk+1];
    uint32_t f2 = (Te0[(s2>>24)&0xff]&0xff000000) ^ (Te0[(s3>>16)&0xff]&0x00ff0000)
                ^ (Te0[(s0>>8)&0xff]&0x0000ff00) ^ (Te0[s1&0xff]&0x000000ff) ^ round_keys[rk+2];
    uint32_t f3 = (Te0[(s3>>24)&0xff]&0xff000000) ^ (Te0[(s0>>16)&0xff]&0x00ff0000)
                ^ (Te0[(s1>>8)&0xff]&0x0000ff00) ^ (Te0[s2&0xff]&0x000000ff) ^ round_keys[rk+3];

    ciphertext[0] = f0;
    ciphertext[1] = f1;
    ciphertext[2] = f2;
    ciphertext[3] = f3;
}

// ------ SHA-256 single-block ------

__device__ static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

extern "C" __global__ void sha256_single_block(
    const uint32_t * __restrict__ msg_words, // 16 words, already big-endian
    uint32_t       * __restrict__ hash_out   // 8 words
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid != 0) return;

    // Initial hash
    uint32_t h0=0x6a09e667, h1=0xbb67ae85, h2=0x3c6ef372, h3=0xa54ff53a;
    uint32_t h4=0x510e527f, h5=0x9b05688c, h6=0x1f83d9ab, h7=0x5be0cd19;

    // Message schedule
    uint32_t W[64];
    for (int i = 0; i < 16; i++) W[i] = msg_words[i];
    for (int i = 16; i < 64; i++) {
        uint32_t s0_ = rotr(W[i-15],7) ^ rotr(W[i-15],18) ^ (W[i-15]>>3);
        uint32_t s1_ = rotr(W[i-2],17) ^ rotr(W[i-2],19)  ^ (W[i-2]>>10);
        W[i] = W[i-16] + s0_ + W[i-7] + s1_;
    }

    uint32_t a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,h=h7;
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t tmp1 = h + S1 + ch + K256[i] + W[i];
        uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t tmp2 = S0 + maj;
        h=g; g=f; f=e; e=d+tmp1; d=c; c=b; b=a; a=tmp1+tmp2;
    }

    hash_out[0]=h0+a; hash_out[1]=h1+b; hash_out[2]=h2+c; hash_out[3]=h3+d;
    hash_out[4]=h4+e; hash_out[5]=h5+f; hash_out[6]=h6+g; hash_out[7]=h7+h;
}
