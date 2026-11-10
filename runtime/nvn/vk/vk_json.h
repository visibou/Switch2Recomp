// A small JSON reader for the shader reflection nxshader writes (<cache>/vk/<key>.json, src/nxshader/nxs_refl.py): a DOM of numbers, strings, arrays
// and objects. Written for this; strings keep ASCII and turn other \u escapes into '?'.
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { VJ_NULL, VJ_BOOL, VJ_NUM, VJ_STR, VJ_ARR, VJ_OBJ };
typedef struct vj {
    int type, n;            // n: elements of an array / members of an object
    double num;             // VJ_NUM, VJ_BOOL (0 / 1)
    char *str;              // VJ_STR
    struct vj *items;       // VJ_ARR, VJ_OBJ (the members' values)
    char **keys;            // VJ_OBJ
} vj;

vj *vj_parse(const char *text, size_t len);   // NULL when the text is not JSON
void vj_free(vj *v);
const vj *vj_get(const vj *obj, const char *key);   // NULL when absent or obj is not an object
const vj *vj_at(const vj *arr, int i);
int vj_int(const vj *v, int dflt);
const char *vj_str(const vj *v, const char *dflt);
int vj_true(const vj *v);

#ifdef __cplusplus
}
#endif
