#ifndef EABITNET_H
#define EABITNET_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Ternary dot product: packed 2-bit weights × int8 activations
// Returns raw dot product (caller applies ternary offset + scale).
int32_t i2_dot_i8(const uint8_t *weights, const int8_t *activations, int32_t n);

// 4-row variant: same activations against 4 weight rows
void i2_dot_i8_4row(
    const uint8_t *w0, const uint8_t *w1,
    const uint8_t *w2, const uint8_t *w3,
    const int8_t *activations,
    int32_t *scores,
    int32_t n
);

#ifdef __cplusplus
}
#endif

#endif // EABITNET_H
