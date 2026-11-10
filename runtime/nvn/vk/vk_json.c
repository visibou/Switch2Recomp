// The JSON reader (vk_json.h): recursive descent over RFC 8259's grammar.
#include "vk_json.h"
#include <stdlib.h>
#include <string.h>

typedef struct { const char *p, *e; int depth; } rd;

static void ws(rd *r) { while (r->p < r->e && (*r->p == ' ' || *r->p == '\t' || *r->p == '\n' || *r->p == '\r')) r->p++; }
static int lit(rd *r, const char *s) { size_t n = strlen(s); if ((size_t)(r->e - r->p) < n || memcmp(r->p, s, n)) return 0; r->p += n; return 1; }

static char *str(rd *r) {
    if (r->p >= r->e || *r->p != '"') return NULL;
    r->p++;
    size_t cap = 16, n = 0; char *o = malloc(cap);
    while (r->p < r->e && *r->p != '"') {
        char c = *r->p++;
        if (c == '\\') {
            if (r->p >= r->e) break;
            char x = *r->p++;
            switch (x) {
            case 'n': c = '\n'; break; case 't': c = '\t'; break; case 'r': c = '\r'; break; case 'b': c = '\b'; break; case 'f': c = '\f'; break;
            case 'u': { unsigned v = 0; for (int k = 0; k < 4 && r->p < r->e; k++) { char h = *r->p++; v = v * 16 + (h >= '0' && h <= '9' ? h - '0' : h >= 'a' && h <= 'f' ? h - 'a' + 10 : h >= 'A' && h <= 'F' ? h - 'A' + 10 : 0); }
                c = v < 128 ? (char)v : '?'; break; }
            default: c = x; break;   // \" \\ \/
            }
        }
        if (n + 1 >= cap) { cap *= 2; o = realloc(o, cap); }
        o[n++] = c;
    }
    if (r->p >= r->e) { free(o); return NULL; }
    r->p++;
    o[n] = 0;
    return o;
}

static int value(rd *r, vj *v);
static int push(vj *v, int *cap) {
    if (v->n == *cap) { *cap = *cap ? 2 * *cap : 4; v->items = realloc(v->items, (size_t)*cap * sizeof *v->items); if (v->type == VJ_OBJ) v->keys = realloc(v->keys, (size_t)*cap * sizeof *v->keys); }
    memset(&v->items[v->n], 0, sizeof v->items[0]);
    return v->n++;
}
static int value(rd *r, vj *v) {
    memset(v, 0, sizeof *v);
    if (++r->depth > 64) return -1;
    ws(r);
    if (r->p >= r->e) return -1;
    int rc = 0;
    if (*r->p == '{') {
        v->type = VJ_OBJ; r->p++; int cap = 0; ws(r);
        if (r->p < r->e && *r->p == '}') r->p++;
        else for (;;) {
            ws(r); char *k = str(r); if (!k) { rc = -1; break; }
            ws(r); if (r->p >= r->e || *r->p != ':') { free(k); rc = -1; break; }
            r->p++;
            int i = push(v, &cap); v->keys[i] = k;
            if (value(r, &v->items[i])) { rc = -1; break; }
            ws(r);
            if (r->p < r->e && *r->p == ',') { r->p++; continue; }
            if (r->p < r->e && *r->p == '}') { r->p++; break; }
            rc = -1; break;
        }
    } else if (*r->p == '[') {
        v->type = VJ_ARR; r->p++; int cap = 0; ws(r);
        if (r->p < r->e && *r->p == ']') r->p++;
        else for (;;) {
            int i = push(v, &cap);
            if (value(r, &v->items[i])) { rc = -1; break; }
            ws(r);
            if (r->p < r->e && *r->p == ',') { r->p++; continue; }
            if (r->p < r->e && *r->p == ']') { r->p++; break; }
            rc = -1; break;
        }
    } else if (*r->p == '"') { v->type = VJ_STR; if (!(v->str = str(r))) rc = -1; }
    else if (lit(r, "true")) { v->type = VJ_BOOL; v->num = 1; }
    else if (lit(r, "false")) { v->type = VJ_BOOL; v->num = 0; }
    else if (lit(r, "null")) v->type = VJ_NULL;
    else {
        char *end; v->type = VJ_NUM; v->num = strtod(r->p, &end);
        if (end == r->p || end > r->e) rc = -1; else r->p = end;
    }
    r->depth--;
    return rc;
}

static void clear(vj *v) {
    if (v->type == VJ_STR) free(v->str);
    if (v->type == VJ_ARR || v->type == VJ_OBJ) {
        for (int i = 0; i < v->n; i++) { clear(&v->items[i]); if (v->type == VJ_OBJ) free(v->keys[i]); }
        free(v->items); free(v->keys);
    }
}
vj *vj_parse(const char *text, size_t len) {
    rd r = {text, text + len, 0};
    vj *v = calloc(1, sizeof *v);
    if (value(&r, v)) { clear(v); free(v); return NULL; }
    return v;
}
void vj_free(vj *v) { if (v) { clear(v); free(v); } }
const vj *vj_get(const vj *o, const char *key) {
    if (!o || o->type != VJ_OBJ) return NULL;
    for (int i = 0; i < o->n; i++) if (!strcmp(o->keys[i], key)) return &o->items[i];
    return NULL;
}
const vj *vj_at(const vj *a, int i) { return a && (a->type == VJ_ARR || a->type == VJ_OBJ) && i >= 0 && i < a->n ? &a->items[i] : NULL; }
int vj_int(const vj *v, int dflt) { return v && (v->type == VJ_NUM || v->type == VJ_BOOL) ? (int)v->num : dflt; }
const char *vj_str(const vj *v, const char *dflt) { return v && v->type == VJ_STR ? v->str : dflt; }
int vj_true(const vj *v) { return v && v->type == VJ_BOOL && v->num != 0; }
