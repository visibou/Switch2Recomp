// A small open-addressing hash map with fixed-size byte keys and pointer values (linear probing, backward-shift deletion). Not thread-safe: callers lock.
#pragma once
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t klen, cap, n; uint8_t *keys; void **vals; } vk_hmap;

// Eight bytes a step (M6 phase B: the replay looks up a ~550-byte pipeline key per draw; a byte a step was 6 % of the render thread)
static inline uint64_t vk_hm_hash(const void *k, uint32_t n) {
    const uint8_t *p = (const uint8_t *)k; uint64_t h = 0x9E3779B97F4A7C15ull ^ n; uint32_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, p + i, 8); h = (h ^ w) * 0x9E3779B97F4A7C15ull; h ^= h >> 29; }
    for (; i < n; i++) { h ^= p[i]; h *= 0x100000001B3ull; }
    h ^= h >> 32; h *= 0xD6E8FEB86659FD93ull;
    return h ^ (h >> 29);
}
static inline uint32_t vk_hm_slot(const vk_hmap *m, const void *k) {
    uint32_t i = (uint32_t)vk_hm_hash(k, m->klen) & (m->cap - 1);
    while (m->vals[i] && memcmp(m->keys + (size_t)i * m->klen, k, m->klen)) i = (i + 1) & (m->cap - 1);
    return i;
}
static inline void *vk_hm_get(const vk_hmap *m, const void *k) { return m->cap ? m->vals[vk_hm_slot(m, k)] : NULL; }
static inline void *vk_hm_put(vk_hmap *m, const void *k, void *v);
static inline void vk_hm_grow(vk_hmap *m) {
    vk_hmap o = *m;
    m->cap = o.cap ? o.cap * 2 : 64; m->n = 0;
    m->keys = (uint8_t *)calloc(m->cap, m->klen); m->vals = (void **)calloc(m->cap, sizeof(void *));
    for (uint32_t i = 0; i < o.cap; i++) if (o.vals[i]) vk_hm_put(m, o.keys + (size_t)i * o.klen, o.vals[i]);
    free(o.keys); free(o.vals);
}
// Returns the value it replaced (NULL if new)
static inline void *vk_hm_put(vk_hmap *m, const void *k, void *v) {
    if ((m->n + 1) * 4 >= m->cap * 3) vk_hm_grow(m);
    uint32_t i = vk_hm_slot(m, k); void *old = m->vals[i];
    if (!old) { memcpy(m->keys + (size_t)i * m->klen, k, m->klen); m->n++; }
    m->vals[i] = v;
    return old;
}
static inline void *vk_hm_del(vk_hmap *m, const void *k) {
    if (!m->cap) return NULL;
    uint32_t i = vk_hm_slot(m, k); void *old = m->vals[i];
    if (!old) return NULL;
    m->vals[i] = NULL; m->n--;
    for (uint32_t j = (i + 1) & (m->cap - 1); m->vals[j]; j = (j + 1) & (m->cap - 1)) {   // move the run back over the hole
        uint32_t home = (uint32_t)vk_hm_hash(m->keys + (size_t)j * m->klen, m->klen) & (m->cap - 1);
        if ((j > i && (home <= i || home > j)) || (j < i && home <= i && home > j)) {
            memcpy(m->keys + (size_t)i * m->klen, m->keys + (size_t)j * m->klen, m->klen); m->vals[i] = m->vals[j]; m->vals[j] = NULL; i = j;
        }
    }
    return old;
}
static inline void vk_hm_free(vk_hmap *m) { free(m->keys); free(m->vals); m->keys = NULL; m->vals = NULL; m->cap = m->n = 0; }
