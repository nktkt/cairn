#include "cairn/cairn.h"
#include "cairn/op_registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#define CAIRN_PATH_MAX 1024
#define CAIRN_BINARY_PLAN_MAGIC "CAIRNPLN"
#define CAIRN_BINARY_PLAN_MAGIC_SIZE 8
#define CAIRN_BINARY_PLAN_VERSION 2
#define CAIRN_BINARY_PLAN_HEADER_SIZE 192
#define CAIRN_BINARY_PLAN_SEGMENT_SIZE 80
#define CAIRN_BINARY_PLAN_TENSOR_SIZE 188
#define CAIRN_BINARY_PLAN_OP_SIZE 124
#define CAIRN_BINARY_PLAN_REF_SIZE 4

typedef struct {
    uint64_t op_id;
    uint64_t bytes;
    uint32_t tick;
    uint32_t dep_first;
    uint32_t dep_count;
    uint32_t input_first;
    uint32_t input_count;
    uint32_t output_first;
    uint32_t output_count;
    cairn_op_class_t op_class;
    char kind[48];
    char stream[32];
} cairn_plan_op_t;

typedef struct {
    uint64_t offset;
    uint64_t nbytes;
    char name[64];
} cairn_memory_segment_t;

typedef struct {
    uint64_t offset;
    uint64_t nbytes;
    uint64_t shape[4];
    uint32_t tensor_id;
    uint32_t dtype_id;
    uint32_t shape_rank;
    char name[64];
    char segment[64];
} cairn_tensor_t;

typedef struct {
    uint64_t tokens;
    uint64_t nbytes;
    char path[CAIRN_DATASET_PATH_MAX];
} cairn_dataset_shard_t;

struct cairn_context {
    cairn_init_desc_t desc;
    char last_error[256];
    char plan_id[65];
    uint64_t step;
    uint64_t token_offset;
    uint64_t op_count;
    uint64_t estimated_memory_bytes;
    uint32_t plan_world_size;
    uint32_t plan_rank;
    uint32_t plan_microbatch_size;
    cairn_plan_op_t *ops;
    cairn_memory_segment_t *segments;
    cairn_tensor_t *tensors;
    uint32_t *dependency_refs;
    uint32_t *tensor_refs;
    cairn_trace_event_t *trace_events;
    uint8_t *scheduler_marks;
    uint64_t *op_logical_end;
    uint64_t segment_count;
    uint64_t tensor_count;
    uint64_t dependency_ref_count;
    uint64_t tensor_ref_count;
    uint64_t trace_count;
    uint64_t trace_capacity;
    cairn_dataset_shard_t *dataset_shards;
    uint64_t dataset_shard_count;
    uint64_t dataset_total_tokens;
    uint32_t dataset_token_bytes;
    int dataset_loaded;
    uint64_t arena_bytes;
    void *arena;
    int arena_allocated;
    cairn_runtime_stats_t stats;
    int plan_loaded;
    int checkpoint_loaded;
    int finalized;
};

static int set_error(cairn_context_t *ctx, int code, const char *message) {
    if (ctx != NULL && message != NULL) {
        snprintf(ctx->last_error, sizeof(ctx->last_error), "%s", message);
    }
    return code;
}

static int file_exists(const char *path) {
    FILE *file;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }
    fclose(file);
    return 1;
}

static int path_is_directory(const char *path) {
    struct stat info;

    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    if (stat(path, &info) != 0) {
        return 0;
    }
    return S_ISDIR(info.st_mode) ? 1 : 0;
}

static int file_size_bytes(const char *path, uint64_t *out_size) {
    struct stat info;

    if (out_size != NULL) {
        *out_size = 0;
    }
    if (path == NULL || path[0] == '\0' || out_size == NULL) {
        return 0;
    }
    if (stat(path, &info) != 0 || !S_ISREG(info.st_mode) || info.st_size < 0) {
        return 0;
    }
    *out_size = (uint64_t)info.st_size;
    return 1;
}

static int ensure_directory(const char *path) {
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    if (mkdir(path, 0777) == 0) {
        return 1;
    }
    if (errno == EEXIST && path_is_directory(path)) {
        return 1;
    }
    return 0;
}

static int join_path(char *out, size_t out_size, const char *left, const char *right) {
    int written;

    if (out == NULL || out_size == 0 || left == NULL || right == NULL || left[0] == '\0' || right[0] == '\0') {
        return 0;
    }
    if (right[0] == '/') {
        written = snprintf(out, out_size, "%s", right);
    } else if (left[strlen(left) - 1] == '/') {
        written = snprintf(out, out_size, "%s%s", left, right);
    } else {
        written = snprintf(out, out_size, "%s/%s", left, right);
    }
    return written > 0 && (size_t)written < out_size;
}

static int resolve_path(char *out, size_t out_size, const char *base_dir, const char *path) {
    if (out == NULL || out_size == 0 || path == NULL || path[0] == '\0') {
        return 0;
    }
    if (path[0] == '/') {
        return snprintf(out, out_size, "%s", path) > 0 && strlen(path) < out_size;
    }
    if (base_dir == NULL || base_dir[0] == '\0') {
        base_dir = ".";
    }
    return join_path(out, out_size, base_dir, path);
}

static int dirname_of(char *out, size_t out_size, const char *path) {
    const char *slash;
    size_t nbytes;

    if (out == NULL || out_size == 0 || path == NULL || path[0] == '\0') {
        return 0;
    }
    slash = strrchr(path, '/');
    if (slash == NULL) {
        if (out_size < 2) {
            return 0;
        }
        snprintf(out, out_size, ".");
        return 1;
    }
    nbytes = (size_t)(slash - path);
    if (nbytes == 0) {
        nbytes = 1;
    }
    if (nbytes + 1 > out_size) {
        return 0;
    }
    memcpy(out, path, nbytes);
    out[nbytes] = '\0';
    return 1;
}

static char *read_file(const char *path, size_t *out_size) {
    FILE *file;
    long size;
    char *buffer;

    if (out_size != NULL) {
        *out_size = 0;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    size = ftell(file);
    if (size < 0) {
        fclose(file);
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }

    buffer = (char *)malloc((size_t)size + 1);
    if (buffer == NULL) {
        fclose(file);
        return NULL;
    }
    if (size > 0 && fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        free(buffer);
        fclose(file);
        return NULL;
    }
    buffer[size] = '\0';
    fclose(file);
    if (out_size != NULL) {
        *out_size = (size_t)size;
    }
    return buffer;
}

static int has_bytes(size_t size, size_t offset, size_t nbytes) {
    return offset <= size && nbytes <= size - offset;
}

static uint32_t read_u32_le(const unsigned char *bytes) {
    return ((uint32_t)bytes[0])
         | ((uint32_t)bytes[1] << 8)
         | ((uint32_t)bytes[2] << 16)
         | ((uint32_t)bytes[3] << 24);
}

static uint64_t read_u64_le(const unsigned char *bytes) {
    return ((uint64_t)bytes[0])
         | ((uint64_t)bytes[1] << 8)
         | ((uint64_t)bytes[2] << 16)
         | ((uint64_t)bytes[3] << 24)
         | ((uint64_t)bytes[4] << 32)
         | ((uint64_t)bytes[5] << 40)
         | ((uint64_t)bytes[6] << 48)
         | ((uint64_t)bytes[7] << 56);
}

static int copy_fixed_ascii(char *out, size_t out_size, const unsigned char *source, size_t source_size) {
    size_t length;
    size_t index;

    if (out == NULL || out_size == 0 || source == NULL || source_size == 0) {
        return 0;
    }

    length = 0;
    while (length < source_size && source[length] != '\0') {
        if (source[length] < 32 || source[length] > 126) {
            return 0;
        }
        length++;
    }
    if (length == 0 || length + 1 > out_size) {
        return 0;
    }
    memcpy(out, source, length);
    out[length] = '\0';

    for (index = length + 1; index < source_size; index++) {
        if (source[index] != '\0') {
            return 0;
        }
    }
    return 1;
}

static const char *find_json_key(const char *json, const char *key) {
    char needle[128];
    if (json == NULL || key == NULL || strlen(key) > 100) {
        return NULL;
    }
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    return strstr(json, needle);
}

static int parse_json_string(const char *json, const char *key, char *out, size_t out_size) {
    const char *cursor;
    size_t index;

    if (out == NULL || out_size == 0) {
        return 0;
    }
    out[0] = '\0';
    cursor = find_json_key(json, key);
    if (cursor == NULL) {
        return 0;
    }
    cursor = strchr(cursor, ':');
    if (cursor == NULL) {
        return 0;
    }
    cursor++;
    while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
        cursor++;
    }
    if (*cursor != '"') {
        return 0;
    }
    cursor++;
    index = 0;
    while (*cursor != '\0' && *cursor != '"') {
        if (*cursor == '\\') {
            return 0;
        }
        if (index + 1 >= out_size) {
            return 0;
        }
        out[index++] = *cursor++;
    }
    if (*cursor != '"') {
        return 0;
    }
    out[index] = '\0';
    return 1;
}

static int parse_json_u64(const char *json, const char *key, uint64_t *out) {
    const char *cursor;
    uint64_t value;

    if (out == NULL) {
        return 0;
    }
    cursor = find_json_key(json, key);
    if (cursor == NULL) {
        return 0;
    }
    cursor = strchr(cursor, ':');
    if (cursor == NULL) {
        return 0;
    }
    cursor++;
    while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
        cursor++;
    }
    if (!isdigit((unsigned char)*cursor)) {
        return 0;
    }
    value = 0;
    while (isdigit((unsigned char)*cursor)) {
        value = (value * 10) + (uint64_t)(*cursor - '0');
        cursor++;
    }
    *out = value;
    return 1;
}

static int parse_json_bool(const char *json, const char *key, int *out) {
    const char *cursor;

    if (out == NULL) {
        return 0;
    }
    cursor = find_json_key(json, key);
    if (cursor == NULL) {
        return 0;
    }
    cursor = strchr(cursor, ':');
    if (cursor == NULL) {
        return 0;
    }
    cursor++;
    while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
        cursor++;
    }
    if (strncmp(cursor, "true", 4) == 0) {
        *out = 1;
        return 1;
    }
    if (strncmp(cursor, "false", 5) == 0) {
        *out = 0;
        return 1;
    }
    return 0;
}

static int atomic_publish(const char *tmp_path, const char *final_path) {
    if (tmp_path == NULL || final_path == NULL) {
        return 0;
    }
    if (rename(tmp_path, final_path) != 0) {
        remove(tmp_path);
        return 0;
    }
    return 1;
}

static uint64_t count_occurrences_range(const char *text, const char *end, const char *needle) {
    uint64_t count;
    size_t needle_len;
    const char *cursor;

    if (text == NULL || end == NULL || end < text || needle == NULL || needle[0] == '\0') {
        return 0;
    }
    count = 0;
    needle_len = strlen(needle);
    cursor = text;
    while ((cursor = strstr(cursor, needle)) != NULL && cursor < end) {
        count++;
        cursor += needle_len;
    }
    return count;
}

static int stream_is_comm(const char *stream) {
    return stream != NULL && strncmp(stream, "comm_", 5) == 0;
}

static int stream_is_io(const char *stream) {
    return stream != NULL && strcmp(stream, "io") == 0;
}

static int stream_is_compute(const char *stream) {
    return stream != NULL && strncmp(stream, "compute_", 8) == 0;
}

static int lookup_executor(const char *kind, cairn_op_class_t *out_class) {
    size_t index;

    if (kind == NULL || out_class == NULL) {
        return 0;
    }
    for (index = 0; index < CAIRN_EXECUTOR_COUNT; index++) {
        if (strcmp(kind, CAIRN_EXECUTORS[index].kind) == 0) {
            *out_class = CAIRN_EXECUTORS[index].op_class;
            return 1;
        }
    }
    return 0;
}

static int stream_matches_class(const char *stream, cairn_op_class_t op_class) {
    if (op_class == CAIRN_OP_CLASS_IO) {
        return stream_is_io(stream);
    }
    if (op_class == CAIRN_OP_CLASS_COMMUNICATION) {
        return stream_is_comm(stream);
    }
    return stream_is_compute(stream);
}

static const char *op_class_name(cairn_op_class_t op_class) {
    if (op_class == CAIRN_OP_CLASS_IO) {
        return "io";
    }
    if (op_class == CAIRN_OP_CLASS_COMMUNICATION) {
        return "communication";
    }
    return "compute";
}

static int dtype_id_is_valid(uint32_t dtype_id) {
    return dtype_id >= 1 && dtype_id <= 7;
}

static uint32_t dtype_id_size_bytes(uint32_t dtype_id) {
    if (dtype_id == 1 || dtype_id == 4 || dtype_id == 7) {
        return 1;
    }
    if (dtype_id == 2 || dtype_id == 6) {
        return 2;
    }
    if (dtype_id == 5) {
        return 4;
    }
    if (dtype_id == 3) {
        return 8;
    }
    return 0;
}

static const cairn_tensor_t *find_tensor_by_name(const cairn_context_t *ctx, const char *name) {
    uint64_t index;

    if (ctx == NULL || name == NULL || name[0] == '\0' || ctx->tensors == NULL) {
        return NULL;
    }
    for (index = 0; index < ctx->tensor_count; index++) {
        if (strcmp(ctx->tensors[index].name, name) == 0) {
            return &ctx->tensors[index];
        }
    }
    return NULL;
}

static uint64_t plan_input_token_count(const cairn_context_t *ctx) {
    const cairn_tensor_t *input_tokens;
    uint32_t dtype_bytes;

    if (ctx == NULL) {
        return 0;
    }
    input_tokens = find_tensor_by_name(ctx, "input_tokens");
    if (input_tokens == NULL) {
        return ctx->plan_microbatch_size;
    }
    dtype_bytes = dtype_id_size_bytes(input_tokens->dtype_id);
    if (dtype_bytes == 0 || input_tokens->nbytes == 0 || input_tokens->nbytes % dtype_bytes != 0) {
        return 0;
    }
    return input_tokens->nbytes / dtype_bytes;
}

static int tensor_arena_range(
    const cairn_context_t *ctx,
    const cairn_tensor_t *tensor,
    uint64_t byte_offset,
    uint64_t nbytes,
    unsigned char **out
) {
    uint64_t arena_offset;

    if (out != NULL) {
        *out = NULL;
    }
    if (ctx == NULL || tensor == NULL || out == NULL || ctx->arena == NULL || !ctx->arena_allocated) {
        return 0;
    }
    if (byte_offset > tensor->nbytes || nbytes > tensor->nbytes - byte_offset) {
        return 0;
    }
    if (UINT64_MAX - tensor->offset < byte_offset) {
        return 0;
    }
    arena_offset = tensor->offset + byte_offset;
    if (arena_offset > ctx->arena_bytes || nbytes > ctx->arena_bytes - arena_offset) {
        return 0;
    }
    if ((uint64_t)((size_t)arena_offset) != arena_offset || (uint64_t)((size_t)nbytes) != nbytes) {
        return 0;
    }
    *out = ((unsigned char *)ctx->arena) + (size_t)arena_offset;
    return 1;
}

static void advance_token_offset(cairn_context_t *ctx, uint64_t token_count) {
    if (ctx == NULL || token_count == 0) {
        return;
    }
    if (ctx->dataset_loaded && ctx->dataset_total_tokens > 0) {
        uint64_t cursor = ctx->token_offset % ctx->dataset_total_tokens;
        uint64_t advance = token_count % ctx->dataset_total_tokens;

        if (ctx->dataset_total_tokens - cursor <= advance) {
            ctx->token_offset = advance - (ctx->dataset_total_tokens - cursor);
        } else {
            ctx->token_offset = cursor + advance;
        }
        return;
    }
    ctx->token_offset += token_count;
}

static int tensor_fits_segment(
    const cairn_tensor_t *tensor,
    const cairn_memory_segment_t *segments,
    uint64_t segment_count
) {
    uint64_t index;

    if (tensor == NULL || segments == NULL || tensor->nbytes == 0) {
        return 0;
    }
    if (UINT64_MAX - tensor->offset < tensor->nbytes) {
        return 0;
    }
    for (index = 0; index < segment_count; index++) {
        uint64_t segment_end;
        uint64_t tensor_end;

        if (strcmp(tensor->segment, segments[index].name) != 0) {
            continue;
        }
        if (UINT64_MAX - segments[index].offset < segments[index].nbytes) {
            return 0;
        }
        segment_end = segments[index].offset + segments[index].nbytes;
        tensor_end = tensor->offset + tensor->nbytes;
        return tensor->offset >= segments[index].offset && tensor_end <= segment_end;
    }
    return 0;
}

static uint32_t dataset_token_bytes_for_dtype(const char *dtype) {
    if (dtype == NULL || dtype[0] == '\0' || strcmp(dtype, "uint32") == 0) {
        return 4;
    }
    if (strcmp(dtype, "uint16") == 0) {
        return 2;
    }
    if (strcmp(dtype, "uint8") == 0) {
        return 1;
    }
    return 0;
}

static const char *find_matching_bracket(const char *open) {
    const char *cursor;
    int depth;
    int in_string;
    int escaped;

    if (open == NULL || *open != '[') {
        return NULL;
    }
    depth = 0;
    in_string = 0;
    escaped = 0;

    for (cursor = open; *cursor != '\0'; cursor++) {
        if (in_string) {
            if (escaped) {
                escaped = 0;
            } else if (*cursor == '\\') {
                escaped = 1;
            } else if (*cursor == '"') {
                in_string = 0;
            }
            continue;
        }

        if (*cursor == '"') {
            in_string = 1;
        } else if (*cursor == '[') {
            depth++;
        } else if (*cursor == ']') {
            depth--;
            if (depth == 0) {
                return cursor;
            }
        }
    }
    return NULL;
}

static const char *find_json_array(const char *json, const char *key, const char **out_end) {
    const char *cursor;
    const char *end;

    if (out_end != NULL) {
        *out_end = NULL;
    }
    cursor = find_json_key(json, key);
    if (cursor == NULL) {
        return NULL;
    }
    cursor = strchr(cursor, ':');
    if (cursor == NULL) {
        return NULL;
    }
    cursor++;
    while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
        cursor++;
    }
    if (*cursor != '[') {
        return NULL;
    }
    end = find_matching_bracket(cursor);
    if (end == NULL) {
        return NULL;
    }
    if (out_end != NULL) {
        *out_end = end;
    }
    return cursor + 1;
}

static const char *find_ops_array(const char *json, const char **out_end) {
    return find_json_array(json, "ops", out_end);
}

static const char *reverse_find_char(const char *start, const char *cursor, char needle) {
    const char *current;

    if (start == NULL || cursor == NULL || cursor < start) {
        return NULL;
    }
    current = cursor;
    while (current >= start) {
        if (*current == needle) {
            return current;
        }
        if (current == start) {
            break;
        }
        current--;
    }
    return NULL;
}

static char *copy_range(const char *start, const char *end) {
    size_t size;
    char *copy;

    if (start == NULL || end == NULL || end < start) {
        return NULL;
    }
    size = (size_t)(end - start) + 1;
    copy = (char *)malloc(size + 1);
    if (copy == NULL) {
        return NULL;
    }
    memcpy(copy, start, size);
    copy[size] = '\0';
    return copy;
}

static int parse_op_table(const char *json, cairn_plan_op_t **out_ops, uint64_t *out_count) {
    const char *ops_start;
    const char *ops_end;
    const char *cursor;
    uint64_t count;
    uint64_t index;
    cairn_plan_op_t *ops;

    if (out_ops == NULL || out_count == NULL) {
        return 0;
    }
    *out_ops = NULL;
    *out_count = 0;

    ops_start = find_ops_array(json, &ops_end);
    if (ops_start == NULL || ops_end == NULL || ops_end <= ops_start) {
        return 0;
    }

    count = count_occurrences_range(ops_start, ops_end, "\"op_id\"");
    if (count == 0) {
        return 0;
    }
    ops = (cairn_plan_op_t *)calloc((size_t)count, sizeof(cairn_plan_op_t));
    if (ops == NULL) {
        return 0;
    }

    cursor = ops_start;
    index = 0;
    while ((cursor = strstr(cursor, "\"op_id\"")) != NULL && cursor < ops_end) {
        const char *object_start;
        const char *object_end;
        char *object_json;
        uint64_t parsed;

        if (index >= count) {
            free(ops);
            return 0;
        }

        object_start = reverse_find_char(ops_start, cursor, '{');
        object_end = strchr(cursor, '}');
        if (object_start == NULL || object_end == NULL || object_end > ops_end) {
            free(ops);
            return 0;
        }
        object_json = copy_range(object_start, object_end);
        if (object_json == NULL) {
            free(ops);
            return 0;
        }

        if (!parse_json_u64(object_json, "op_id", &parsed) || parsed != index) {
            free(object_json);
            free(ops);
            return 0;
        }
        ops[index].op_id = parsed;
        ops[index].tick = (uint32_t)parsed;
        if (!parse_json_string(object_json, "kind", ops[index].kind, sizeof(ops[index].kind))) {
            free(object_json);
            free(ops);
            return 0;
        }
        if (!lookup_executor(ops[index].kind, &ops[index].op_class)) {
            free(object_json);
            free(ops);
            return 0;
        }
        if (!parse_json_string(object_json, "stream", ops[index].stream, sizeof(ops[index].stream))) {
            free(object_json);
            free(ops);
            return 0;
        }
        if (!stream_matches_class(ops[index].stream, ops[index].op_class)) {
            free(object_json);
            free(ops);
            return 0;
        }
        if (parse_json_u64(object_json, "bytes", &parsed)) {
            ops[index].bytes = parsed;
        } else {
            ops[index].bytes = 0;
        }

        free(object_json);
        index++;
        cursor = object_end + 1;
    }

    if (index != count) {
        free(ops);
        return 0;
    }
    *out_ops = ops;
    *out_count = count;
    return 1;
}

static int parse_memory_segments(
    const char *json,
    cairn_memory_segment_t **out_segments,
    uint64_t *out_count,
    uint64_t *out_arena_bytes
) {
    const char *segments_start;
    const char *segments_end;
    const char *cursor;
    uint64_t count;
    uint64_t index;
    uint64_t previous_end;
    uint64_t arena_bytes;
    cairn_memory_segment_t *segments;

    if (out_segments == NULL || out_count == NULL || out_arena_bytes == NULL) {
        return 0;
    }
    *out_segments = NULL;
    *out_count = 0;
    *out_arena_bytes = 0;

    segments_start = find_json_array(json, "segments", &segments_end);
    if (segments_start == NULL || segments_end == NULL || segments_end <= segments_start) {
        return 0;
    }

    count = count_occurrences_range(segments_start, segments_end, "\"name\"");
    if (count == 0) {
        return 0;
    }
    segments = (cairn_memory_segment_t *)calloc((size_t)count, sizeof(cairn_memory_segment_t));
    if (segments == NULL) {
        return 0;
    }

    cursor = segments_start;
    index = 0;
    previous_end = 0;
    arena_bytes = 0;
    while ((cursor = strstr(cursor, "\"name\"")) != NULL && cursor < segments_end) {
        const char *object_start;
        const char *object_end;
        char *object_json;
        uint64_t offset;
        uint64_t nbytes;

        if (index >= count) {
            free(segments);
            return 0;
        }

        object_start = reverse_find_char(segments_start, cursor, '{');
        object_end = strchr(cursor, '}');
        if (object_start == NULL || object_end == NULL || object_end > segments_end) {
            free(segments);
            return 0;
        }
        object_json = copy_range(object_start, object_end);
        if (object_json == NULL) {
            free(segments);
            return 0;
        }

        if (!parse_json_string(object_json, "name", segments[index].name, sizeof(segments[index].name))) {
            free(object_json);
            free(segments);
            return 0;
        }
        if (!parse_json_u64(object_json, "offset", &offset)) {
            free(object_json);
            free(segments);
            return 0;
        }
        if (!parse_json_u64(object_json, "nbytes", &nbytes) || nbytes == 0) {
            free(object_json);
            free(segments);
            return 0;
        }
        if (offset < previous_end || UINT64_MAX - offset < nbytes) {
            free(object_json);
            free(segments);
            return 0;
        }

        segments[index].offset = offset;
        segments[index].nbytes = nbytes;
        previous_end = offset + nbytes;
        if (previous_end > arena_bytes) {
            arena_bytes = previous_end;
        }

        free(object_json);
        index++;
        cursor = object_end + 1;
    }

    if (index != count || arena_bytes == 0) {
        free(segments);
        return 0;
    }

    *out_segments = segments;
    *out_count = count;
    *out_arena_bytes = arena_bytes;
    return 1;
}

static int parse_dataset_shards(
    const char *json,
    const char *manifest_dir,
    uint32_t token_bytes,
    cairn_dataset_shard_t **out_shards,
    uint64_t *out_count,
    uint64_t *out_total_tokens
) {
    const char *shards_start;
    const char *shards_end;
    const char *cursor;
    uint64_t count;
    uint64_t index;
    uint64_t total_tokens;
    cairn_dataset_shard_t *shards;

    if (out_shards == NULL || out_count == NULL || out_total_tokens == NULL || token_bytes == 0) {
        return 0;
    }
    *out_shards = NULL;
    *out_count = 0;
    *out_total_tokens = 0;

    shards_start = find_json_array(json, "shards", &shards_end);
    if (shards_start == NULL || shards_end == NULL || shards_end <= shards_start) {
        return 0;
    }

    count = count_occurrences_range(shards_start, shards_end, "\"path\"");
    if (count == 0 || (uint64_t)((size_t)count) != count) {
        return 0;
    }
    shards = (cairn_dataset_shard_t *)calloc((size_t)count, sizeof(cairn_dataset_shard_t));
    if (shards == NULL) {
        return 0;
    }

    cursor = shards_start;
    index = 0;
    total_tokens = 0;
    while ((cursor = strstr(cursor, "\"path\"")) != NULL && cursor < shards_end) {
        const char *object_start;
        const char *object_end;
        char *object_json;
        char relative_path[CAIRN_DATASET_PATH_MAX];
        uint64_t expected_nbytes;
        uint64_t actual_nbytes;
        uint64_t tokens;

        if (index >= count) {
            free(shards);
            return 0;
        }

        object_start = reverse_find_char(shards_start, cursor, '{');
        object_end = strchr(cursor, '}');
        if (object_start == NULL || object_end == NULL || object_end > shards_end) {
            free(shards);
            return 0;
        }
        object_json = copy_range(object_start, object_end);
        if (object_json == NULL) {
            free(shards);
            return 0;
        }

        if (!parse_json_string(object_json, "path", relative_path, sizeof(relative_path))) {
            free(object_json);
            free(shards);
            return 0;
        }
        if (!resolve_path(shards[index].path, sizeof(shards[index].path), manifest_dir, relative_path)) {
            free(object_json);
            free(shards);
            return 0;
        }
        if (!parse_json_u64(object_json, "tokens", &tokens) || tokens == 0) {
            free(object_json);
            free(shards);
            return 0;
        }
        if (tokens > UINT64_MAX / token_bytes) {
            free(object_json);
            free(shards);
            return 0;
        }
        expected_nbytes = tokens * token_bytes;
        if (!file_size_bytes(shards[index].path, &actual_nbytes) || actual_nbytes != expected_nbytes) {
            free(object_json);
            free(shards);
            return 0;
        }
        if (UINT64_MAX - total_tokens < tokens) {
            free(object_json);
            free(shards);
            return 0;
        }

        shards[index].tokens = tokens;
        shards[index].nbytes = actual_nbytes;
        total_tokens += tokens;
        free(object_json);
        index++;
        cursor = object_end + 1;
    }

    if (index != count || total_tokens == 0) {
        free(shards);
        return 0;
    }

    *out_shards = shards;
    *out_count = count;
    *out_total_tokens = total_tokens;
    return 1;
}

static int resolve_dataset_cursor(
    const cairn_context_t *ctx,
    uint64_t token_offset,
    uint32_t *out_shard_index,
    uint64_t *out_shard_token_offset,
    uint64_t *out_tokens_available,
    char *out_path,
    size_t out_path_size
) {
    uint64_t cursor;
    uint64_t shard_index;

    if (out_shard_index != NULL) {
        *out_shard_index = 0;
    }
    if (out_shard_token_offset != NULL) {
        *out_shard_token_offset = token_offset;
    }
    if (out_tokens_available != NULL) {
        *out_tokens_available = UINT64_MAX;
    }
    if (out_path != NULL && out_path_size > 0) {
        out_path[0] = '\0';
    }

    if (ctx == NULL || !ctx->dataset_loaded) {
        return 1;
    }
    if (ctx->dataset_total_tokens == 0 || ctx->dataset_shard_count == 0 || ctx->dataset_shards == NULL) {
        return 0;
    }

    cursor = token_offset % ctx->dataset_total_tokens;
    for (shard_index = 0; shard_index < ctx->dataset_shard_count; shard_index++) {
        const cairn_dataset_shard_t *shard = &ctx->dataset_shards[shard_index];

        if (cursor < shard->tokens) {
            if (out_shard_index != NULL) {
                *out_shard_index = (uint32_t)shard_index;
            }
            if (out_shard_token_offset != NULL) {
                *out_shard_token_offset = cursor;
            }
            if (out_tokens_available != NULL) {
                *out_tokens_available = shard->tokens - cursor;
            }
            if (out_path != NULL && out_path_size > 0) {
                snprintf(out_path, out_path_size, "%s", shard->path);
            }
            return 1;
        }
        cursor -= shard->tokens;
    }
    return 0;
}

static int reserve_arena(cairn_context_t *ctx, uint64_t arena_bytes) {
    const char *allocate_host_arena;

    if (ctx == NULL || arena_bytes == 0) {
        return 0;
    }

    allocate_host_arena = getenv("CAIRN_ALLOCATE_HOST_ARENA");
    if (allocate_host_arena != NULL && strcmp(allocate_host_arena, "1") == 0) {
        if ((uint64_t)((size_t)arena_bytes) != arena_bytes) {
            return 0;
        }
        ctx->arena = malloc((size_t)arena_bytes);
        if (ctx->arena == NULL) {
            return 0;
        }
        ctx->arena_allocated = 1;
    } else {
        ctx->arena = malloc(1);
        if (ctx->arena == NULL) {
            return 0;
        }
        ctx->arena_allocated = 0;
    }
    ctx->arena_bytes = arena_bytes;
    return 1;
}

static int reserve_execution_state(cairn_context_t *ctx, uint64_t op_count) {
    if (ctx == NULL || op_count == 0 || (uint64_t)((size_t)op_count) != op_count) {
        return 0;
    }

    ctx->trace_events = (cairn_trace_event_t *)calloc((size_t)op_count, sizeof(cairn_trace_event_t));
    ctx->scheduler_marks = (uint8_t *)calloc((size_t)op_count, sizeof(uint8_t));
    ctx->op_logical_end = (uint64_t *)calloc((size_t)op_count, sizeof(uint64_t));
    if (ctx->trace_events == NULL || ctx->scheduler_marks == NULL || ctx->op_logical_end == NULL) {
        free(ctx->trace_events);
        free(ctx->scheduler_marks);
        free(ctx->op_logical_end);
        ctx->trace_events = NULL;
        ctx->scheduler_marks = NULL;
        ctx->op_logical_end = NULL;
        ctx->trace_capacity = 0;
        ctx->trace_count = 0;
        return 0;
    }
    ctx->trace_capacity = op_count;
    ctx->trace_count = 0;
    return 1;
}

static int restore_checkpoint_contents(cairn_context_t *ctx, const char *contents) {
    char checkpoint_plan_id[65];
    uint64_t parsed;
    uint64_t checkpoint_token_offset;
    uint64_t expected_shard_offset;
    uint64_t expected_tokens_available;
    uint32_t expected_shard_index;
    int checkpoint_dataset_loaded;
    int has_checkpoint_dataset_loaded;

    if (ctx == NULL || contents == NULL) {
        return 0;
    }
    checkpoint_token_offset = ctx->token_offset;
    expected_shard_index = 0;
    expected_shard_offset = 0;
    expected_tokens_available = 0;
    checkpoint_dataset_loaded = 0;
    has_checkpoint_dataset_loaded = 0;
    if (parse_json_string(contents, "plan_id", checkpoint_plan_id, sizeof(checkpoint_plan_id))) {
        if (ctx->plan_loaded && strcmp(checkpoint_plan_id, ctx->plan_id) != 0) {
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint plan_id does not match loaded plan");
        }
    }
    if (parse_json_u64(contents, "step", &parsed)) {
        ctx->step = parsed;
    }
    if (parse_json_u64(contents, "token_offset", &parsed)) {
        ctx->token_offset = parsed;
        checkpoint_token_offset = parsed;
    }
    if (parse_json_bool(contents, "dataset_loaded", &checkpoint_dataset_loaded)) {
        has_checkpoint_dataset_loaded = 1;
    }
    if (has_checkpoint_dataset_loaded && checkpoint_dataset_loaded) {
        if (!ctx->dataset_loaded) {
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint requires dataset manifest before restore");
        }
        if (!parse_json_u64(contents, "dataset_shard_count", &parsed) || parsed != ctx->dataset_shard_count) {
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint dataset shard count does not match loaded dataset");
        }
        if (!parse_json_u64(contents, "dataset_total_tokens", &parsed) || parsed != ctx->dataset_total_tokens) {
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint dataset token count does not match loaded dataset");
        }
        if (!parse_json_u64(contents, "dataset_token_bytes", &parsed) || parsed != ctx->dataset_token_bytes) {
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint dataset token dtype does not match loaded dataset");
        }
        if (!resolve_dataset_cursor(
                ctx,
                checkpoint_token_offset,
                &expected_shard_index,
                &expected_shard_offset,
                &expected_tokens_available,
                NULL,
                0)) {
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint dataset cursor could not be resolved");
        }
        if (!parse_json_u64(contents, "dataset_cursor_shard_index", &parsed) || parsed != expected_shard_index) {
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint dataset shard cursor does not match token offset");
        }
        if (!parse_json_u64(contents, "dataset_cursor_shard_token_offset", &parsed) || parsed != expected_shard_offset) {
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint dataset token cursor does not match token offset");
        }
        if (!parse_json_u64(contents, "dataset_cursor_tokens_available", &parsed) || parsed != expected_tokens_available) {
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint dataset available token count does not match token offset");
        }
    }
    if (parse_json_u64(contents, "ops_executed", &parsed)) {
        ctx->stats.ops_executed = parsed;
    }
    if (parse_json_u64(contents, "compute_ops", &parsed)) {
        ctx->stats.compute_ops = parsed;
    }
    if (parse_json_u64(contents, "communication_ops", &parsed)) {
        ctx->stats.communication_ops = parsed;
    }
    if (parse_json_u64(contents, "io_ops", &parsed)) {
        ctx->stats.io_ops = parsed;
    }
    if (parse_json_u64(contents, "compute_bytes", &parsed)) {
        ctx->stats.compute_bytes = parsed;
    }
    if (parse_json_u64(contents, "communication_bytes", &parsed)) {
        ctx->stats.communication_bytes = parsed;
    }
    if (parse_json_u64(contents, "io_bytes", &parsed)) {
        ctx->stats.io_bytes = parsed;
    }
    if (ctx->op_count > 0 && ctx->stats.ops_executed >= ctx->op_count) {
        ctx->stats.steps_executed = ctx->stats.ops_executed / ctx->op_count;
    }
    return CAIRN_OK;
}

static int write_rank_checkpoint(cairn_context_t *ctx, const char *path) {
    FILE *file;
    char tmp_path[CAIRN_PATH_MAX];
    uint32_t dataset_cursor_shard_index;
    uint64_t dataset_cursor_shard_token_offset;
    uint64_t dataset_cursor_tokens_available;

    if (snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path) <= 0 || strlen(path) + 4 >= sizeof(tmp_path)) {
        return set_error(ctx, CAIRN_ERR_IO, "rank checkpoint temporary path is too long");
    }
    dataset_cursor_shard_index = 0;
    dataset_cursor_shard_token_offset = 0;
    dataset_cursor_tokens_available = 0;
    if (ctx->dataset_loaded) {
        if (!resolve_dataset_cursor(
                ctx,
                ctx->token_offset,
                &dataset_cursor_shard_index,
                &dataset_cursor_shard_token_offset,
                &dataset_cursor_tokens_available,
                NULL,
                0)) {
            return set_error(ctx, CAIRN_ERR_STATE, "dataset cursor could not be resolved for checkpoint");
        }
    }

    file = fopen(tmp_path, "wb");
    if (file == NULL) {
        return set_error(ctx, CAIRN_ERR_IO, "rank checkpoint shard could not be opened for writing");
    }
    fprintf(file, "{\n");
    fprintf(file, "  \"version\": 1,\n");
    fprintf(file, "  \"plan_id\": \"%s\",\n", ctx->plan_id);
    fprintf(file, "  \"global_rank\": %u,\n", ctx->desc.global_rank);
    fprintf(file, "  \"world_size\": %u,\n", ctx->plan_world_size);
    fprintf(file, "  \"step\": %llu,\n", (unsigned long long)ctx->step);
    fprintf(file, "  \"token_offset\": %llu,\n", (unsigned long long)ctx->token_offset);
    fprintf(file, "  \"op_count\": %llu,\n", (unsigned long long)ctx->op_count);
    fprintf(file, "  \"estimated_memory_bytes\": %llu,\n", (unsigned long long)ctx->estimated_memory_bytes);
    fprintf(file, "  \"memory_segment_count\": %llu,\n", (unsigned long long)ctx->segment_count);
    fprintf(file, "  \"tensor_count\": %llu,\n", (unsigned long long)ctx->tensor_count);
    fprintf(file, "  \"dependency_ref_count\": %llu,\n", (unsigned long long)ctx->dependency_ref_count);
    fprintf(file, "  \"tensor_ref_count\": %llu,\n", (unsigned long long)ctx->tensor_ref_count);
    fprintf(file, "  \"dataset_loaded\": %s,\n", ctx->dataset_loaded ? "true" : "false");
    fprintf(file, "  \"dataset_shard_count\": %llu,\n", (unsigned long long)ctx->dataset_shard_count);
    fprintf(file, "  \"dataset_total_tokens\": %llu,\n", (unsigned long long)ctx->dataset_total_tokens);
    fprintf(file, "  \"dataset_token_bytes\": %u,\n", ctx->dataset_token_bytes);
    fprintf(file, "  \"dataset_cursor_shard_index\": %u,\n", dataset_cursor_shard_index);
    fprintf(file, "  \"dataset_cursor_shard_token_offset\": %llu,\n", (unsigned long long)dataset_cursor_shard_token_offset);
    fprintf(file, "  \"dataset_cursor_tokens_available\": %llu,\n", (unsigned long long)dataset_cursor_tokens_available);
    fprintf(file, "  \"arena_bytes\": %llu,\n", (unsigned long long)ctx->arena_bytes);
    fprintf(file, "  \"arena_allocated\": %s,\n", ctx->arena_allocated ? "true" : "false");
    fprintf(file, "  \"ops_executed\": %llu,\n", (unsigned long long)ctx->stats.ops_executed);
    fprintf(file, "  \"compute_ops\": %llu,\n", (unsigned long long)ctx->stats.compute_ops);
    fprintf(file, "  \"communication_ops\": %llu,\n", (unsigned long long)ctx->stats.communication_ops);
    fprintf(file, "  \"io_ops\": %llu,\n", (unsigned long long)ctx->stats.io_ops);
    fprintf(file, "  \"trace_event_count\": %llu,\n", (unsigned long long)ctx->trace_count);
    fprintf(file, "  \"compute_bytes\": %llu,\n", (unsigned long long)ctx->stats.compute_bytes);
    fprintf(file, "  \"communication_bytes\": %llu,\n", (unsigned long long)ctx->stats.communication_bytes);
    fprintf(file, "  \"io_bytes\": %llu\n", (unsigned long long)ctx->stats.io_bytes);
    fprintf(file, "}\n");
    if (fclose(file) != 0) {
        remove(tmp_path);
        return set_error(ctx, CAIRN_ERR_IO, "rank checkpoint shard could not be closed");
    }
    if (!atomic_publish(tmp_path, path)) {
        return set_error(ctx, CAIRN_ERR_IO, "rank checkpoint shard could not be published");
    }
    return CAIRN_OK;
}

static void clear_plan(cairn_context_t *ctx) {
    if (ctx == NULL) {
        return;
    }
    free(ctx->ops);
    ctx->ops = NULL;
    free(ctx->segments);
    ctx->segments = NULL;
    free(ctx->tensors);
    ctx->tensors = NULL;
    free(ctx->dependency_refs);
    ctx->dependency_refs = NULL;
    free(ctx->tensor_refs);
    ctx->tensor_refs = NULL;
    free(ctx->trace_events);
    ctx->trace_events = NULL;
    free(ctx->scheduler_marks);
    ctx->scheduler_marks = NULL;
    free(ctx->op_logical_end);
    ctx->op_logical_end = NULL;
    free(ctx->arena);
    ctx->arena = NULL;
    ctx->op_count = 0;
    ctx->segment_count = 0;
    ctx->tensor_count = 0;
    ctx->dependency_ref_count = 0;
    ctx->tensor_ref_count = 0;
    ctx->trace_count = 0;
    ctx->trace_capacity = 0;
    ctx->estimated_memory_bytes = 0;
    ctx->arena_bytes = 0;
    ctx->arena_allocated = 0;
    ctx->plan_world_size = 0;
    ctx->plan_rank = 0;
    ctx->plan_microbatch_size = 1;
    ctx->plan_id[0] = '\0';
    memset(&ctx->stats, 0, sizeof(ctx->stats));
    ctx->plan_loaded = 0;
}

static void clear_dataset(cairn_context_t *ctx) {
    if (ctx == NULL) {
        return;
    }
    free(ctx->dataset_shards);
    ctx->dataset_shards = NULL;
    ctx->dataset_shard_count = 0;
    ctx->dataset_total_tokens = 0;
    ctx->dataset_token_bytes = 0;
    ctx->dataset_loaded = 0;
}

static int fail_binary_plan(
    cairn_context_t *ctx,
    cairn_plan_op_t *ops,
    cairn_memory_segment_t *segments,
    cairn_tensor_t *tensors,
    uint32_t *dependency_refs,
    uint32_t *tensor_refs,
    const char *message
) {
    free(ops);
    free(segments);
    free(tensors);
    free(dependency_refs);
    free(tensor_refs);
    return set_error(ctx, CAIRN_ERR_PLAN, message);
}

static int load_binary_rank_plan(cairn_context_t *ctx, const unsigned char *contents, size_t contents_size) {
    size_t offset;
    size_t expected_size;
    uint32_t binary_version;
    uint32_t registry_version;
    uint32_t loaded_world_size;
    uint32_t loaded_rank;
    uint32_t loaded_microbatch_size;
    uint32_t loaded_op_count;
    uint32_t loaded_segment_count;
    uint32_t loaded_tensor_count;
    uint32_t loaded_dependency_ref_count;
    uint32_t loaded_tensor_ref_count;
    uint64_t loaded_memory_bytes;
    uint64_t loaded_arena_bytes;
    uint64_t computed_arena_bytes;
    uint64_t previous_end;
    char loaded_registry_sha[65];
    char loaded_plan_id[65];
    cairn_plan_op_t *loaded_ops;
    cairn_memory_segment_t *loaded_segments;
    cairn_tensor_t *loaded_tensors;
    uint32_t *loaded_dependency_refs;
    uint32_t *loaded_tensor_refs;
    uint32_t index;

    if (ctx == NULL || contents == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (!has_bytes(contents_size, 0, CAIRN_BINARY_PLAN_HEADER_SIZE)) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan header is incomplete");
    }
    if (memcmp(contents, CAIRN_BINARY_PLAN_MAGIC, CAIRN_BINARY_PLAN_MAGIC_SIZE) != 0) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan magic is invalid");
    }

    offset = CAIRN_BINARY_PLAN_MAGIC_SIZE;
    binary_version = read_u32_le(contents + offset);
    offset += 4;
    registry_version = read_u32_le(contents + offset);
    offset += 4;
    if (!copy_fixed_ascii(loaded_registry_sha, sizeof(loaded_registry_sha), contents + offset, 64)) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan registry hash is invalid");
    }
    offset += 64;
    if (!copy_fixed_ascii(loaded_plan_id, sizeof(loaded_plan_id), contents + offset, 64)) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan plan_id is invalid");
    }
    offset += 64;
    loaded_world_size = read_u32_le(contents + offset);
    offset += 4;
    loaded_rank = read_u32_le(contents + offset);
    offset += 4;
    loaded_microbatch_size = read_u32_le(contents + offset);
    offset += 4;
    loaded_op_count = read_u32_le(contents + offset);
    offset += 4;
    loaded_segment_count = read_u32_le(contents + offset);
    offset += 4;
    loaded_tensor_count = read_u32_le(contents + offset);
    offset += 4;
    loaded_dependency_ref_count = read_u32_le(contents + offset);
    offset += 4;
    loaded_tensor_ref_count = read_u32_le(contents + offset);
    offset += 4;
    loaded_memory_bytes = read_u64_le(contents + offset);
    offset += 8;
    loaded_arena_bytes = read_u64_le(contents + offset);
    offset += 8;

    if (offset != CAIRN_BINARY_PLAN_HEADER_SIZE) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan header size mismatch");
    }
    if (binary_version != CAIRN_BINARY_PLAN_VERSION) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan version does not match runtime");
    }
    if (registry_version != CAIRN_OP_REGISTRY_VERSION) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan op registry version does not match runtime");
    }
    if (strcmp(loaded_registry_sha, CAIRN_OP_REGISTRY_SHA256) != 0) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan op registry hash does not match runtime");
    }
    if (loaded_world_size == 0 || loaded_world_size != ctx->desc.world_size) {
        return set_error(ctx, CAIRN_ERR_PLAN, "runtime world_size does not match binary rank plan");
    }
    if (loaded_rank != ctx->desc.global_rank) {
        return set_error(ctx, CAIRN_ERR_PLAN, "runtime global_rank does not match binary rank plan");
    }
    if (loaded_microbatch_size == 0) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan microbatch_size is invalid");
    }
    if (loaded_op_count == 0 || loaded_segment_count == 0 || loaded_tensor_count == 0 || loaded_arena_bytes == 0) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan has empty op, tensor, or memory tables");
    }
    expected_size = CAIRN_BINARY_PLAN_HEADER_SIZE + ((size_t)loaded_segment_count * CAIRN_BINARY_PLAN_SEGMENT_SIZE);
    if ((size_t)loaded_tensor_count > (((size_t)-1) - expected_size) / CAIRN_BINARY_PLAN_TENSOR_SIZE) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan tensor table is too large");
    }
    expected_size += (size_t)loaded_tensor_count * CAIRN_BINARY_PLAN_TENSOR_SIZE;
    if ((size_t)loaded_op_count > (((size_t)-1) - expected_size) / CAIRN_BINARY_PLAN_OP_SIZE) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan op table is too large");
    }
    expected_size += (size_t)loaded_op_count * CAIRN_BINARY_PLAN_OP_SIZE;
    if ((size_t)loaded_dependency_ref_count > (((size_t)-1) - expected_size) / CAIRN_BINARY_PLAN_REF_SIZE) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan dependency table is too large");
    }
    expected_size += (size_t)loaded_dependency_ref_count * CAIRN_BINARY_PLAN_REF_SIZE;
    if ((size_t)loaded_tensor_ref_count > (((size_t)-1) - expected_size) / CAIRN_BINARY_PLAN_REF_SIZE) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan tensor reference table is too large");
    }
    expected_size += (size_t)loaded_tensor_ref_count * CAIRN_BINARY_PLAN_REF_SIZE;
    if (contents_size != expected_size) {
        return set_error(ctx, CAIRN_ERR_PLAN, "binary rank plan size does not match header");
    }

    loaded_ops = NULL;
    loaded_segments = NULL;
    loaded_tensors = NULL;
    loaded_dependency_refs = NULL;
    loaded_tensor_refs = NULL;
    loaded_segments = (cairn_memory_segment_t *)calloc((size_t)loaded_segment_count, sizeof(cairn_memory_segment_t));
    if (loaded_segments == NULL) {
        return set_error(ctx, CAIRN_ERR_STATE, "binary rank plan memory segments could not be allocated");
    }
    loaded_tensors = (cairn_tensor_t *)calloc((size_t)loaded_tensor_count, sizeof(cairn_tensor_t));
    if (loaded_tensors == NULL) {
        free(loaded_segments);
        return set_error(ctx, CAIRN_ERR_STATE, "binary rank plan tensor table could not be allocated");
    }
    loaded_ops = (cairn_plan_op_t *)calloc((size_t)loaded_op_count, sizeof(cairn_plan_op_t));
    if (loaded_ops == NULL) {
        free(loaded_tensors);
        free(loaded_segments);
        return set_error(ctx, CAIRN_ERR_STATE, "binary rank plan op table could not be allocated");
    }
    loaded_dependency_refs = (uint32_t *)calloc((size_t)loaded_dependency_ref_count, sizeof(uint32_t));
    if (loaded_dependency_ref_count > 0 && loaded_dependency_refs == NULL) {
        free(loaded_ops);
        free(loaded_tensors);
        free(loaded_segments);
        return set_error(ctx, CAIRN_ERR_STATE, "binary rank plan dependency table could not be allocated");
    }
    loaded_tensor_refs = (uint32_t *)calloc((size_t)loaded_tensor_ref_count, sizeof(uint32_t));
    if (loaded_tensor_ref_count > 0 && loaded_tensor_refs == NULL) {
        free(loaded_dependency_refs);
        free(loaded_ops);
        free(loaded_tensors);
        free(loaded_segments);
        return set_error(ctx, CAIRN_ERR_STATE, "binary rank plan tensor reference table could not be allocated");
    }

    previous_end = 0;
    computed_arena_bytes = 0;
    for (index = 0; index < loaded_segment_count; index++) {
        uint64_t segment_end;

        if (!copy_fixed_ascii(loaded_segments[index].name, sizeof(loaded_segments[index].name), contents + offset, 64)) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan memory segment name is invalid");
        }
        offset += 64;
        loaded_segments[index].offset = read_u64_le(contents + offset);
        offset += 8;
        loaded_segments[index].nbytes = read_u64_le(contents + offset);
        offset += 8;
        if (loaded_segments[index].nbytes == 0) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan memory segment has zero size");
        }
        if (loaded_segments[index].offset < previous_end) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan memory segments overlap");
        }
        if (UINT64_MAX - loaded_segments[index].offset < loaded_segments[index].nbytes) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan memory segment overflows");
        }
        segment_end = loaded_segments[index].offset + loaded_segments[index].nbytes;
        previous_end = segment_end;
        if (segment_end > computed_arena_bytes) {
            computed_arena_bytes = segment_end;
        }
    }
    if (computed_arena_bytes != loaded_arena_bytes) {
        return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan arena size does not match segments");
    }

    for (index = 0; index < loaded_tensor_count; index++) {
        uint32_t dim_index;

        loaded_tensors[index].tensor_id = read_u32_le(contents + offset);
        offset += 4;
        loaded_tensors[index].dtype_id = read_u32_le(contents + offset);
        offset += 4;
        if (loaded_tensors[index].tensor_id != index) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan tensor ids are not contiguous");
        }
        if (!dtype_id_is_valid(loaded_tensors[index].dtype_id)) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan tensor dtype is invalid");
        }
        if (!copy_fixed_ascii(loaded_tensors[index].name, sizeof(loaded_tensors[index].name), contents + offset, 64)) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan tensor name is invalid");
        }
        offset += 64;
        if (!copy_fixed_ascii(loaded_tensors[index].segment, sizeof(loaded_tensors[index].segment), contents + offset, 64)) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan tensor segment is invalid");
        }
        offset += 64;
        loaded_tensors[index].offset = read_u64_le(contents + offset);
        offset += 8;
        loaded_tensors[index].nbytes = read_u64_le(contents + offset);
        offset += 8;
        loaded_tensors[index].shape_rank = read_u32_le(contents + offset);
        offset += 4;
        if (loaded_tensors[index].shape_rank > 4) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan tensor shape rank is invalid");
        }
        for (dim_index = 0; dim_index < 4; dim_index++) {
            loaded_tensors[index].shape[dim_index] = read_u64_le(contents + offset);
            offset += 8;
        }
        if (!tensor_fits_segment(&loaded_tensors[index], loaded_segments, loaded_segment_count)) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan tensor does not fit its memory segment");
        }
    }

    for (index = 0; index < loaded_op_count; index++) {
        uint32_t loaded_op_id;
        uint32_t loaded_class;
        cairn_op_class_t registry_class;

        loaded_op_id = read_u32_le(contents + offset);
        offset += 4;
        loaded_class = read_u32_le(contents + offset);
        offset += 4;
        if (loaded_op_id != index) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op ids are not contiguous");
        }
        if (loaded_class > (uint32_t)CAIRN_OP_CLASS_IO) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op class is invalid");
        }
        loaded_ops[index].op_id = loaded_op_id;
        loaded_ops[index].op_class = (cairn_op_class_t)loaded_class;
        if (!copy_fixed_ascii(loaded_ops[index].kind, sizeof(loaded_ops[index].kind), contents + offset, 48)) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op kind is invalid");
        }
        offset += 48;
        if (!copy_fixed_ascii(loaded_ops[index].stream, sizeof(loaded_ops[index].stream), contents + offset, 32)) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op stream is invalid");
        }
        offset += 32;
        loaded_ops[index].bytes = read_u64_le(contents + offset);
        offset += 8;
        loaded_ops[index].tick = read_u32_le(contents + offset);
        offset += 4;
        loaded_ops[index].dep_first = read_u32_le(contents + offset);
        offset += 4;
        loaded_ops[index].dep_count = read_u32_le(contents + offset);
        offset += 4;
        loaded_ops[index].input_first = read_u32_le(contents + offset);
        offset += 4;
        loaded_ops[index].input_count = read_u32_le(contents + offset);
        offset += 4;
        loaded_ops[index].output_first = read_u32_le(contents + offset);
        offset += 4;
        loaded_ops[index].output_count = read_u32_le(contents + offset);
        offset += 4;

        if (!lookup_executor(loaded_ops[index].kind, &registry_class)) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op table contains unsupported op kind");
        }
        if (registry_class != loaded_ops[index].op_class) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op class does not match registry");
        }
        if (!stream_matches_class(loaded_ops[index].stream, loaded_ops[index].op_class)) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op stream does not match op class");
        }
        if (index > 0 && loaded_ops[index].tick < loaded_ops[index - 1].tick) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op ticks are not monotonic");
        }
        if (loaded_ops[index].dep_first > loaded_dependency_ref_count
            || loaded_ops[index].dep_count > loaded_dependency_ref_count - loaded_ops[index].dep_first) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op dependency range is invalid");
        }
        if (loaded_ops[index].input_first > loaded_tensor_ref_count
            || loaded_ops[index].input_count > loaded_tensor_ref_count - loaded_ops[index].input_first) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op input tensor range is invalid");
        }
        if (loaded_ops[index].output_first > loaded_tensor_ref_count
            || loaded_ops[index].output_count > loaded_tensor_ref_count - loaded_ops[index].output_first) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op output tensor range is invalid");
        }
        if (loaded_ops[index].input_count == 0) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan op has no input tensors");
        }
        if (loaded_ops[index].op_class != CAIRN_OP_CLASS_IO && loaded_ops[index].output_count == 0) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan non-io op has no output tensors");
        }
    }

    for (index = 0; index < loaded_dependency_ref_count; index++) {
        loaded_dependency_refs[index] = read_u32_le(contents + offset);
        offset += 4;
        if (loaded_dependency_refs[index] >= loaded_op_count) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan dependency references invalid op");
        }
    }
    for (index = 0; index < loaded_op_count; index++) {
        uint32_t dep_index;
        for (dep_index = 0; dep_index < loaded_ops[index].dep_count; dep_index++) {
            uint32_t dep = loaded_dependency_refs[loaded_ops[index].dep_first + dep_index];
            if (dep >= index) {
                return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan dependencies are not acyclic");
            }
        }
        if (index == 0 && loaded_ops[index].dep_count != 0) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan first op must not have dependencies");
        }
        if (index > 0 && loaded_ops[index].dep_count == 0) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan non-first op has no dependencies");
        }
    }
    for (index = 0; index < loaded_tensor_ref_count; index++) {
        loaded_tensor_refs[index] = read_u32_le(contents + offset);
        offset += 4;
        if (loaded_tensor_refs[index] >= loaded_tensor_count) {
            return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan tensor reference is invalid");
        }
    }
    if (offset != contents_size) {
        return fail_binary_plan(ctx, loaded_ops, loaded_segments, loaded_tensors, loaded_dependency_refs, loaded_tensor_refs, "binary rank plan trailing bytes are invalid");
    }

    clear_plan(ctx);
    snprintf(ctx->plan_id, sizeof(ctx->plan_id), "%s", loaded_plan_id);
    ctx->plan_world_size = loaded_world_size;
    ctx->plan_rank = loaded_rank;
    ctx->plan_microbatch_size = loaded_microbatch_size;
    ctx->estimated_memory_bytes = loaded_memory_bytes;
    ctx->ops = loaded_ops;
    ctx->op_count = loaded_op_count;
    ctx->segments = loaded_segments;
    ctx->segment_count = loaded_segment_count;
    ctx->tensors = loaded_tensors;
    ctx->tensor_count = loaded_tensor_count;
    ctx->dependency_refs = loaded_dependency_refs;
    ctx->dependency_ref_count = loaded_dependency_ref_count;
    ctx->tensor_refs = loaded_tensor_refs;
    ctx->tensor_ref_count = loaded_tensor_ref_count;
    if (!reserve_arena(ctx, loaded_arena_bytes)) {
        clear_plan(ctx);
        return set_error(ctx, CAIRN_ERR_STATE, "memory arena could not be reserved");
    }
    if (!reserve_execution_state(ctx, ctx->op_count)) {
        clear_plan(ctx);
        return set_error(ctx, CAIRN_ERR_STATE, "runtime execution state could not be reserved");
    }
    ctx->plan_loaded = 1;
    return CAIRN_OK;
}

int cairn_init(cairn_context_t **ctx, const cairn_init_desc_t *desc) {
    cairn_context_t *created;

    if (ctx == NULL || desc == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (desc->world_size == 0 || desc->global_rank >= desc->world_size) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }

    created = (cairn_context_t *)calloc(1, sizeof(cairn_context_t));
    if (created == NULL) {
        return CAIRN_ERR_STATE;
    }

    created->desc = *desc;
    created->plan_id[0] = '\0';
    created->step = 0;
    created->token_offset = 0;
    created->op_count = 0;
    created->estimated_memory_bytes = 0;
    created->plan_world_size = 0;
    created->plan_rank = 0;
    created->plan_microbatch_size = 1;
    created->ops = NULL;
    created->segments = NULL;
    created->tensors = NULL;
    created->dependency_refs = NULL;
    created->tensor_refs = NULL;
    created->trace_events = NULL;
    created->scheduler_marks = NULL;
    created->op_logical_end = NULL;
    created->segment_count = 0;
    created->tensor_count = 0;
    created->dependency_ref_count = 0;
    created->tensor_ref_count = 0;
    created->trace_count = 0;
    created->trace_capacity = 0;
    created->dataset_shards = NULL;
    created->dataset_shard_count = 0;
    created->dataset_total_tokens = 0;
    created->dataset_token_bytes = 0;
    created->dataset_loaded = 0;
    created->arena_bytes = 0;
    created->arena = NULL;
    created->arena_allocated = 0;
    memset(&created->stats, 0, sizeof(created->stats));
    created->plan_loaded = 0;
    created->checkpoint_loaded = 0;
    created->finalized = 0;
    created->last_error[0] = '\0';
    *ctx = created;
    return CAIRN_OK;
}

int cairn_load_plan(cairn_context_t *ctx, const char *path) {
    char *contents;
    size_t contents_size;
    uint64_t parsed;
    char loaded_plan_id[65];
    uint32_t loaded_world_size;
    uint32_t loaded_rank;
    uint32_t loaded_microbatch_size;
    uint64_t loaded_memory_bytes;
    uint64_t loaded_op_count;
    uint64_t loaded_segment_count;
    uint64_t loaded_arena_bytes;
    cairn_plan_op_t *loaded_ops;
    cairn_memory_segment_t *loaded_segments;
    char loaded_registry_sha[65];

    if (ctx == NULL || path == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!file_exists(path)) {
        return set_error(ctx, CAIRN_ERR_IO, "plan file does not exist");
    }

    contents = read_file(path, &contents_size);
    if (contents == NULL || contents_size == 0) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_IO, "plan file could not be read");
    }
    if (contents_size >= CAIRN_BINARY_PLAN_MAGIC_SIZE
        && memcmp(contents, CAIRN_BINARY_PLAN_MAGIC, CAIRN_BINARY_PLAN_MAGIC_SIZE) == 0) {
        int status = load_binary_rank_plan(ctx, (const unsigned char *)contents, contents_size);
        free(contents);
        return status;
    }
    loaded_plan_id[0] = '\0';
    loaded_world_size = ctx->desc.world_size;
    loaded_rank = ctx->desc.global_rank;
    loaded_microbatch_size = 1;
    loaded_memory_bytes = 0;
    loaded_op_count = 0;
    loaded_segment_count = 0;
    loaded_arena_bytes = 0;
    loaded_ops = NULL;
    loaded_segments = NULL;
    loaded_registry_sha[0] = '\0';

    if (!parse_json_string(contents, "plan_id", loaded_plan_id, sizeof(loaded_plan_id))) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_PLAN, "plan file does not contain plan_id");
    }
    if (!parse_json_u64(contents, "op_registry_version", &parsed) || parsed != CAIRN_OP_REGISTRY_VERSION) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_PLAN, "plan op registry version does not match runtime");
    }
    if (!parse_json_string(contents, "op_registry_sha256", loaded_registry_sha, sizeof(loaded_registry_sha))) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_PLAN, "plan op registry hash is missing");
    }
    if (strcmp(loaded_registry_sha, CAIRN_OP_REGISTRY_SHA256) != 0) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_PLAN, "plan op registry hash does not match runtime");
    }
    if (parse_json_u64(contents, "world_size", &parsed)) {
        if (parsed == 0 || parsed > UINT32_MAX) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "plan world_size is invalid");
        }
        loaded_world_size = (uint32_t)parsed;
        if (ctx->desc.world_size != 0 && ctx->desc.world_size != loaded_world_size) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "runtime world_size does not match plan world_size");
        }
    }
    if (parse_json_u64(contents, "global_rank", &parsed)) {
        if (parsed > UINT32_MAX) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "plan global_rank is invalid");
        }
        loaded_rank = (uint32_t)parsed;
        if (loaded_rank != ctx->desc.global_rank) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "runtime global_rank does not match rank plan");
        }
    }
    if (parse_json_u64(contents, "microbatch_size", &parsed)) {
        if (parsed == 0 || parsed > UINT32_MAX) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "plan microbatch_size is invalid");
        }
        loaded_microbatch_size = (uint32_t)parsed;
    }
    if (parse_json_u64(contents, "estimated_bytes_per_rank", &parsed)) {
        loaded_memory_bytes = parsed;
    }
    if (!parse_op_table(contents, &loaded_ops, &loaded_op_count)) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_PLAN, "plan op table is missing or invalid");
    }
    if (!parse_memory_segments(contents, &loaded_segments, &loaded_segment_count, &loaded_arena_bytes)) {
        free(loaded_ops);
        free(contents);
        return set_error(ctx, CAIRN_ERR_PLAN, "plan memory segments are missing or invalid");
    }
    free(contents);

    clear_plan(ctx);
    snprintf(ctx->plan_id, sizeof(ctx->plan_id), "%s", loaded_plan_id);
    ctx->plan_world_size = loaded_world_size;
    ctx->plan_rank = loaded_rank;
    ctx->plan_microbatch_size = loaded_microbatch_size;
    ctx->estimated_memory_bytes = loaded_memory_bytes;
    ctx->ops = loaded_ops;
    ctx->op_count = loaded_op_count;
    ctx->segments = loaded_segments;
    ctx->segment_count = loaded_segment_count;
    if (!reserve_arena(ctx, loaded_arena_bytes)) {
        clear_plan(ctx);
        return set_error(ctx, CAIRN_ERR_STATE, "memory arena could not be reserved");
    }
    if (!reserve_execution_state(ctx, ctx->op_count)) {
        clear_plan(ctx);
        return set_error(ctx, CAIRN_ERR_STATE, "runtime execution state could not be reserved");
    }
    ctx->plan_loaded = 1;
    return CAIRN_OK;
}

int cairn_load_dataset(cairn_context_t *ctx, const char *path) {
    char *contents;
    size_t contents_size;
    char manifest_dir[CAIRN_PATH_MAX];
    char format[32];
    char token_dtype[32];
    uint64_t parsed;
    uint32_t token_bytes;
    cairn_dataset_shard_t *loaded_shards;
    uint64_t loaded_shard_count;
    uint64_t loaded_total_tokens;

    if (ctx == NULL || path == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!file_exists(path)) {
        return set_error(ctx, CAIRN_ERR_IO, "dataset manifest file does not exist");
    }

    contents = read_file(path, &contents_size);
    if (contents == NULL || contents_size == 0) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_IO, "dataset manifest could not be read");
    }
    if (!parse_json_u64(contents, "version", &parsed) || parsed != 1) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_PLAN, "dataset manifest version is invalid");
    }
    if (!parse_json_string(contents, "format", format, sizeof(format)) || strcmp(format, "fixed-token-binary") != 0) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_PLAN, "dataset format must be fixed-token-binary");
    }
    if (!parse_json_string(contents, "token_dtype", token_dtype, sizeof(token_dtype))) {
        snprintf(token_dtype, sizeof(token_dtype), "uint32");
    }
    token_bytes = dataset_token_bytes_for_dtype(token_dtype);
    if (token_bytes == 0) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_PLAN, "dataset token_dtype is unsupported");
    }
    if (!dirname_of(manifest_dir, sizeof(manifest_dir), path)) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_IO, "dataset manifest directory is invalid");
    }

    loaded_shards = NULL;
    loaded_shard_count = 0;
    loaded_total_tokens = 0;
    if (!parse_dataset_shards(
            contents,
            manifest_dir,
            token_bytes,
            &loaded_shards,
            &loaded_shard_count,
            &loaded_total_tokens)) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_PLAN, "dataset shards are missing, invalid, or do not match binary size");
    }
    free(contents);

    clear_dataset(ctx);
    ctx->dataset_shards = loaded_shards;
    ctx->dataset_shard_count = loaded_shard_count;
    ctx->dataset_total_tokens = loaded_total_tokens;
    ctx->dataset_token_bytes = token_bytes;
    ctx->dataset_loaded = 1;
    return CAIRN_OK;
}

int cairn_load_checkpoint(cairn_context_t *ctx, const char *path) {
    char *contents;
    size_t contents_size;
    char latest_path[CAIRN_PATH_MAX];
    char manifest_rel[CAIRN_PATH_MAX];
    char manifest_path[CAIRN_PATH_MAX];
    char manifest_dir[CAIRN_PATH_MAX];
    char rank_shard_rel[CAIRN_PATH_MAX];
    char rank_shard_path[CAIRN_PATH_MAX];
    int complete;
    int status;

    if (ctx == NULL || path == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!ctx->dataset_loaded && ctx->desc.dataset_manifest_path != NULL && ctx->desc.dataset_manifest_path[0] != '\0') {
        status = cairn_load_dataset(ctx, ctx->desc.dataset_manifest_path);
        if (status != CAIRN_OK) {
            return status;
        }
    }
    if (path_is_directory(path)) {
        if (!join_path(latest_path, sizeof(latest_path), path, "latest.json")) {
            return set_error(ctx, CAIRN_ERR_IO, "checkpoint latest path is too long");
        }
        contents = read_file(latest_path, &contents_size);
        if (contents == NULL || contents_size == 0) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_IO, "checkpoint latest manifest could not be read");
        }
        if (!parse_json_string(contents, "manifest_path", manifest_rel, sizeof(manifest_rel))) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint latest manifest does not contain manifest_path");
        }
        if (!parse_json_bool(contents, "complete", &complete) || !complete) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint latest manifest is incomplete");
        }
        free(contents);

        if (!join_path(manifest_path, sizeof(manifest_path), path, manifest_rel)) {
            return set_error(ctx, CAIRN_ERR_IO, "checkpoint manifest path is too long");
        }
        contents = read_file(manifest_path, &contents_size);
        if (contents == NULL || contents_size == 0) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_IO, "checkpoint manifest could not be read");
        }
        if (!parse_json_string(contents, "rank_shard_path", rank_shard_rel, sizeof(rank_shard_rel))) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint manifest does not contain rank_shard_path");
        }
        if (!parse_json_bool(contents, "complete", &complete) || !complete) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint manifest is incomplete");
        }
        if (!dirname_of(manifest_dir, sizeof(manifest_dir), manifest_path)) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_IO, "checkpoint manifest directory is invalid");
        }
        free(contents);

        if (!join_path(rank_shard_path, sizeof(rank_shard_path), manifest_dir, rank_shard_rel)) {
            return set_error(ctx, CAIRN_ERR_IO, "checkpoint rank shard path is too long");
        }
        contents = read_file(rank_shard_path, &contents_size);
    } else {
        contents = read_file(path, &contents_size);
    }
    if (contents == NULL || contents_size == 0) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint file could not be read");
    }
    status = restore_checkpoint_contents(ctx, contents);
    free(contents);
    if (status != CAIRN_OK) {
        return status;
    }
    ctx->checkpoint_loaded = 1;
    return CAIRN_OK;
}

int cairn_next_batch(cairn_context_t *ctx, cairn_batch_t *batch) {
    if (ctx == NULL || batch == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!ctx->plan_loaded) {
        return set_error(ctx, CAIRN_ERR_STATE, "plan must be loaded before requesting batches");
    }
    if (!ctx->dataset_loaded && ctx->desc.dataset_manifest_path != NULL && ctx->desc.dataset_manifest_path[0] != '\0') {
        int status = cairn_load_dataset(ctx, ctx->desc.dataset_manifest_path);
        if (status != CAIRN_OK) {
            return status;
        }
    }

    batch->step = ctx->step;
    batch->token_offset = ctx->token_offset;
    batch->microbatch_size = ctx->plan_microbatch_size;
    if (!resolve_dataset_cursor(
            ctx,
            ctx->token_offset,
            &batch->shard_index,
            &batch->shard_token_offset,
            &batch->tokens_available,
            batch->shard_path,
            sizeof(batch->shard_path))) {
        return set_error(ctx, CAIRN_ERR_STATE, "dataset cursor could not be resolved to a shard");
    }
    return CAIRN_OK;
}

static int read_tokens_from_shard(
    const cairn_dataset_shard_t *shard,
    uint64_t shard_token_offset,
    uint32_t token_bytes,
    uint64_t token_count,
    unsigned char *out
) {
    FILE *file;
    uint64_t byte_offset;
    uint64_t nbytes;

    if (shard == NULL || out == NULL || token_bytes == 0 || token_count == 0) {
        return 0;
    }
    if (shard_token_offset > shard->tokens || token_count > shard->tokens - shard_token_offset) {
        return 0;
    }
    if (shard_token_offset > UINT64_MAX / token_bytes || token_count > UINT64_MAX / token_bytes) {
        return 0;
    }
    byte_offset = shard_token_offset * token_bytes;
    nbytes = token_count * token_bytes;
    if ((uint64_t)((long)byte_offset) != byte_offset || (uint64_t)((size_t)nbytes) != nbytes) {
        return 0;
    }

    file = fopen(shard->path, "rb");
    if (file == NULL) {
        return 0;
    }
    if (fseek(file, (long)byte_offset, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    if (fread(out, 1, (size_t)nbytes, file) != (size_t)nbytes) {
        fclose(file);
        return 0;
    }
    fclose(file);
    return 1;
}

int cairn_read_batch_tokens(
    cairn_context_t *ctx,
    const cairn_batch_t *batch,
    void *out,
    uint64_t token_count,
    uint64_t out_nbytes,
    uint64_t *out_tokens_read
) {
    unsigned char *cursor_out;
    uint64_t remaining;
    uint64_t current_token_offset;
    uint64_t bytes_required;
    uint64_t tokens_read;

    if (out_tokens_read != NULL) {
        *out_tokens_read = 0;
    }
    if (ctx == NULL || batch == NULL || out == NULL || token_count == 0) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!ctx->dataset_loaded || ctx->dataset_total_tokens == 0 || ctx->dataset_shard_count == 0) {
        return set_error(ctx, CAIRN_ERR_STATE, "dataset must be loaded before reading batch tokens");
    }
    if (ctx->dataset_token_bytes == 0 || token_count > UINT64_MAX / ctx->dataset_token_bytes) {
        return set_error(ctx, CAIRN_ERR_STATE, "dataset token byte count is invalid");
    }
    bytes_required = token_count * ctx->dataset_token_bytes;
    if (out_nbytes < bytes_required || (uint64_t)((size_t)bytes_required) != bytes_required) {
        return set_error(ctx, CAIRN_ERR_INVALID_ARGUMENT, "batch token output buffer is too small");
    }

    cursor_out = (unsigned char *)out;
    remaining = token_count;
    current_token_offset = batch->token_offset % ctx->dataset_total_tokens;
    tokens_read = 0;
    while (remaining > 0) {
        uint32_t shard_index;
        uint64_t shard_token_offset;
        uint64_t tokens_available;
        uint64_t to_read;

        if (!resolve_dataset_cursor(
                ctx,
                current_token_offset,
                &shard_index,
                &shard_token_offset,
                &tokens_available,
                NULL,
                0)) {
            return set_error(ctx, CAIRN_ERR_STATE, "dataset cursor could not be resolved while reading tokens");
        }
        if (shard_index >= ctx->dataset_shard_count || tokens_available == 0) {
            return set_error(ctx, CAIRN_ERR_STATE, "dataset shard cursor is invalid while reading tokens");
        }
        to_read = remaining < tokens_available ? remaining : tokens_available;
        if (!read_tokens_from_shard(
                &ctx->dataset_shards[shard_index],
                shard_token_offset,
                ctx->dataset_token_bytes,
                to_read,
                cursor_out)) {
            return set_error(ctx, CAIRN_ERR_IO, "dataset shard token bytes could not be read");
        }
        cursor_out += (size_t)(to_read * ctx->dataset_token_bytes);
        remaining -= to_read;
        tokens_read += to_read;
        if (ctx->dataset_total_tokens - current_token_offset <= to_read) {
            current_token_offset = to_read - (ctx->dataset_total_tokens - current_token_offset);
        } else {
            current_token_offset += to_read;
        }
    }

    if (out_tokens_read != NULL) {
        *out_tokens_read = tokens_read;
    }
    return CAIRN_OK;
}

int cairn_stage_batch_input(cairn_context_t *ctx, const cairn_batch_t *batch, uint64_t *out_tokens_staged) {
    const cairn_tensor_t *input_tokens;
    unsigned char *destination;
    uint64_t token_count;
    uint64_t tokens_read;
    uint32_t input_dtype_bytes;
    int status;

    if (out_tokens_staged != NULL) {
        *out_tokens_staged = 0;
    }
    if (ctx == NULL || batch == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!ctx->plan_loaded) {
        return set_error(ctx, CAIRN_ERR_STATE, "plan must be loaded before staging batch input");
    }
    if (!ctx->arena_allocated || ctx->arena == NULL) {
        return set_error(ctx, CAIRN_ERR_STATE, "host arena is metadata-only; set CAIRN_ALLOCATE_HOST_ARENA=1");
    }
    if (!ctx->dataset_loaded || ctx->dataset_token_bytes == 0) {
        return set_error(ctx, CAIRN_ERR_STATE, "dataset must be loaded before staging batch input");
    }
    input_tokens = find_tensor_by_name(ctx, "input_tokens");
    if (input_tokens == NULL) {
        return set_error(ctx, CAIRN_ERR_PLAN, "plan does not contain input_tokens tensor metadata");
    }
    input_dtype_bytes = dtype_id_size_bytes(input_tokens->dtype_id);
    if (input_dtype_bytes == 0 || input_dtype_bytes != ctx->dataset_token_bytes) {
        return set_error(ctx, CAIRN_ERR_PLAN, "input_tokens tensor dtype does not match dataset token dtype");
    }
    if (input_tokens->nbytes == 0 || input_tokens->nbytes % input_dtype_bytes != 0) {
        return set_error(ctx, CAIRN_ERR_PLAN, "input_tokens tensor byte size is invalid");
    }
    if (!tensor_arena_range(ctx, input_tokens, 0, input_tokens->nbytes, &destination)) {
        return set_error(ctx, CAIRN_ERR_STATE, "input_tokens tensor is not writable in the host arena");
    }

    token_count = input_tokens->nbytes / input_dtype_bytes;
    status = cairn_read_batch_tokens(ctx, batch, destination, token_count, input_tokens->nbytes, &tokens_read);
    if (status != CAIRN_OK) {
        return status;
    }
    if (tokens_read != token_count) {
        return set_error(ctx, CAIRN_ERR_IO, "dataset reader staged fewer input tokens than requested");
    }
    if (out_tokens_staged != NULL) {
        *out_tokens_staged = tokens_read;
    }
    return CAIRN_OK;
}

int cairn_copy_tensor_bytes(
    cairn_context_t *ctx,
    const char *tensor_name,
    uint64_t byte_offset,
    void *out,
    uint64_t out_nbytes
) {
    const cairn_tensor_t *tensor;
    unsigned char *source;

    if (ctx == NULL || tensor_name == NULL || tensor_name[0] == '\0' || out == NULL || out_nbytes == 0) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!ctx->plan_loaded) {
        return set_error(ctx, CAIRN_ERR_STATE, "plan must be loaded before copying tensor bytes");
    }
    if (!ctx->arena_allocated || ctx->arena == NULL) {
        return set_error(ctx, CAIRN_ERR_STATE, "host arena is metadata-only; set CAIRN_ALLOCATE_HOST_ARENA=1");
    }
    tensor = find_tensor_by_name(ctx, tensor_name);
    if (tensor == NULL) {
        return set_error(ctx, CAIRN_ERR_PLAN, "tensor name is not present in the loaded plan");
    }
    if (!tensor_arena_range(ctx, tensor, byte_offset, out_nbytes, &source)) {
        return set_error(ctx, CAIRN_ERR_INVALID_ARGUMENT, "tensor byte range is outside the host arena allocation");
    }
    memcpy(out, source, (size_t)out_nbytes);
    return CAIRN_OK;
}

static void update_stats_for_op(cairn_context_t *ctx, const cairn_plan_op_t *op) {
    ctx->stats.ops_executed += 1;
    if (op->op_class == CAIRN_OP_CLASS_IO) {
        ctx->stats.io_ops += 1;
        ctx->stats.io_bytes += op->bytes;
    } else if (op->op_class == CAIRN_OP_CLASS_COMMUNICATION) {
        ctx->stats.communication_ops += 1;
        ctx->stats.communication_bytes += op->bytes;
    } else {
        ctx->stats.compute_ops += 1;
        ctx->stats.compute_bytes += op->bytes;
    }
}

static void record_trace_event(
    cairn_context_t *ctx,
    const cairn_plan_op_t *op,
    uint64_t ordinal,
    uint64_t logical_start,
    uint64_t logical_end
) {
    cairn_trace_event_t *event;

    if (ctx == NULL || op == NULL || ctx->trace_events == NULL || ctx->trace_count >= ctx->trace_capacity) {
        return;
    }

    event = &ctx->trace_events[ctx->trace_count++];
    memset(event, 0, sizeof(*event));
    event->step = ctx->step;
    event->ordinal = ordinal;
    event->bytes = op->bytes;
    event->logical_start = logical_start;
    event->logical_end = logical_end;
    event->op_id = (uint32_t)op->op_id;
    event->tick = op->tick;
    event->dep_count = op->dep_count;
    event->input_count = op->input_count;
    event->output_count = op->output_count;
    event->op_class = (uint32_t)op->op_class;
    snprintf(event->kind, sizeof(event->kind), "%s", op->kind);
    snprintf(event->stream, sizeof(event->stream), "%s", op->stream);
}

static int op_dependencies_ready(const cairn_context_t *ctx, uint64_t op_index, uint64_t *logical_start) {
    const cairn_plan_op_t *op;
    uint32_t dep_index;
    uint64_t start;

    if (ctx == NULL || logical_start == NULL || op_index >= ctx->op_count) {
        return 0;
    }

    op = &ctx->ops[op_index];
    start = op->tick;
    for (dep_index = 0; dep_index < op->dep_count; dep_index++) {
        uint32_t dep = ctx->dependency_refs[op->dep_first + dep_index];

        if (dep >= ctx->op_count || ctx->scheduler_marks[dep] == 0) {
            return 0;
        }
        if (ctx->op_logical_end[dep] > start) {
            start = ctx->op_logical_end[dep];
        }
    }
    *logical_start = start;
    return 1;
}

static int execute_op(cairn_context_t *ctx, uint64_t op_index, uint64_t ordinal, uint64_t logical_start) {
    const cairn_plan_op_t *op;
    uint64_t logical_end;

    if (ctx == NULL || op_index >= ctx->op_count) {
        return 0;
    }

    op = &ctx->ops[op_index];
    logical_end = logical_start + 1;
    update_stats_for_op(ctx, op);
    if (ctx->op_logical_end != NULL) {
        ctx->op_logical_end[op_index] = logical_end;
    }
    record_trace_event(ctx, op, ordinal, logical_start, logical_end);
    return 1;
}

static int execute_sequential_plan(cairn_context_t *ctx) {
    uint64_t index;

    for (index = 0; index < ctx->op_count; index++) {
        const cairn_plan_op_t *op = &ctx->ops[index];
        uint64_t logical_start = op->tick > index ? op->tick : index;

        if (!execute_op(ctx, index, index, logical_start)) {
            return 0;
        }
    }
    return 1;
}

static int execute_dependency_plan(cairn_context_t *ctx) {
    uint64_t executed;

    if (ctx == NULL || ctx->scheduler_marks == NULL || ctx->op_logical_end == NULL) {
        return 0;
    }

    memset(ctx->scheduler_marks, 0, (size_t)ctx->op_count);
    memset(ctx->op_logical_end, 0, (size_t)ctx->op_count * sizeof(uint64_t));
    executed = 0;
    while (executed < ctx->op_count) {
        uint64_t index;
        int made_progress = 0;

        for (index = 0; index < ctx->op_count; index++) {
            uint64_t logical_start;

            if (ctx->scheduler_marks[index] != 0) {
                continue;
            }
            if (!op_dependencies_ready(ctx, index, &logical_start)) {
                continue;
            }
            if (!execute_op(ctx, index, executed, logical_start)) {
                return 0;
            }
            ctx->scheduler_marks[index] = 1;
            executed++;
            made_progress = 1;
        }
        if (!made_progress) {
            return 0;
        }
    }
    return 1;
}

int cairn_train_step(cairn_context_t *ctx, const cairn_batch_t *batch) {
    uint64_t input_token_count;
    int stage_status;

    if (ctx == NULL || batch == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!ctx->plan_loaded) {
        return set_error(ctx, CAIRN_ERR_STATE, "plan must be loaded before training");
    }
    if (batch->step != ctx->step) {
        return set_error(ctx, CAIRN_ERR_STATE, "batch step does not match context step");
    }
    if (ctx->ops == NULL || ctx->op_count == 0) {
        return set_error(ctx, CAIRN_ERR_PLAN, "loaded plan has no executable ops");
    }
    input_token_count = plan_input_token_count(ctx);
    if (input_token_count == 0) {
        return set_error(ctx, CAIRN_ERR_PLAN, "plan input token count is invalid");
    }
    if (ctx->dataset_loaded && ctx->arena_allocated && find_tensor_by_name(ctx, "input_tokens") != NULL) {
        stage_status = cairn_stage_batch_input(ctx, batch, NULL);
        if (stage_status != CAIRN_OK) {
            return stage_status;
        }
    }

    ctx->trace_count = 0;
    if (ctx->dependency_ref_count > 0) {
        if (!execute_dependency_plan(ctx)) {
            return set_error(ctx, CAIRN_ERR_PLAN, "dependency scheduler could not execute loaded op graph");
        }
    } else if (!execute_sequential_plan(ctx)) {
        return set_error(ctx, CAIRN_ERR_PLAN, "sequential scheduler could not execute loaded op table");
    }
    ctx->stats.steps_executed += 1;
    ctx->step += 1;
    advance_token_offset(ctx, input_token_count);
    return CAIRN_OK;
}

int cairn_get_stats(const cairn_context_t *ctx, cairn_runtime_stats_t *stats) {
    if (ctx == NULL || stats == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    *stats = ctx->stats;
    return CAIRN_OK;
}

uint64_t cairn_trace_event_count(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->trace_count;
}

int cairn_get_trace_event(const cairn_context_t *ctx, uint64_t index, cairn_trace_event_t *event) {
    if (ctx == NULL || event == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (index >= ctx->trace_count || ctx->trace_events == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    *event = ctx->trace_events[index];
    return CAIRN_OK;
}

int cairn_write_trace(cairn_context_t *ctx, const char *path) {
    FILE *file;
    char tmp_path[CAIRN_PATH_MAX];
    uint64_t index;

    if (ctx == NULL || path == NULL || path[0] == '\0') {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!ctx->plan_loaded) {
        return set_error(ctx, CAIRN_ERR_STATE, "plan must be loaded before writing trace");
    }
    if (snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path) <= 0 || strlen(path) + 4 >= sizeof(tmp_path)) {
        return set_error(ctx, CAIRN_ERR_IO, "trace temporary path is too long");
    }

    file = fopen(tmp_path, "wb");
    if (file == NULL) {
        return set_error(ctx, CAIRN_ERR_IO, "trace file could not be opened for writing");
    }
    for (index = 0; index < ctx->trace_count; index++) {
        const cairn_trace_event_t *event = &ctx->trace_events[index];

        fprintf(file,
                "{\"version\":1,\"plan_id\":\"%s\",\"global_rank\":%u,\"step\":%llu,\"ordinal\":%llu,\"op_id\":%u,\"tick\":%u,\"kind\":\"%s\",\"stream\":\"%s\",\"op_class\":\"%s\",\"dep_count\":%u,\"input_count\":%u,\"output_count\":%u,\"bytes\":%llu,\"logical_start\":%llu,\"logical_end\":%llu}\n",
                ctx->plan_id,
                ctx->desc.global_rank,
                (unsigned long long)event->step,
                (unsigned long long)event->ordinal,
                event->op_id,
                event->tick,
                event->kind,
                event->stream,
                op_class_name((cairn_op_class_t)event->op_class),
                event->dep_count,
                event->input_count,
                event->output_count,
                (unsigned long long)event->bytes,
                (unsigned long long)event->logical_start,
                (unsigned long long)event->logical_end);
    }
    if (fclose(file) != 0) {
        remove(tmp_path);
        return set_error(ctx, CAIRN_ERR_IO, "trace file could not be closed");
    }
    if (!atomic_publish(tmp_path, path)) {
        return set_error(ctx, CAIRN_ERR_IO, "trace file could not be published");
    }
    return CAIRN_OK;
}

int cairn_should_checkpoint(cairn_context_t *ctx) {
    if (ctx == NULL || ctx->finalized || ctx->step == 0) {
        return 0;
    }
    return (ctx->step % 100) == 0;
}

int cairn_save_checkpoint(cairn_context_t *ctx, const char *tag) {
    FILE *file;
    char step_name[64];
    char step_dir[CAIRN_PATH_MAX];
    char ranks_dir[CAIRN_PATH_MAX];
    char rank_file_name[64];
    char rank_path[CAIRN_PATH_MAX];
    char manifest_path[CAIRN_PATH_MAX];
    char latest_path[CAIRN_PATH_MAX];
    char manifest_rel[CAIRN_PATH_MAX];
    char manifest_tmp[CAIRN_PATH_MAX];
    char latest_tmp[CAIRN_PATH_MAX];
    int status;

    if (ctx == NULL || tag == NULL || tag[0] == '\0') {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!ctx->plan_loaded) {
        return set_error(ctx, CAIRN_ERR_STATE, "plan must be loaded before checkpointing");
    }

    if (!ensure_directory(tag)) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint root directory could not be created");
    }
    snprintf(step_name, sizeof(step_name), "step_%09llu", (unsigned long long)ctx->step);
    if (!join_path(step_dir, sizeof(step_dir), tag, step_name)) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint step path is too long");
    }
    if (!ensure_directory(step_dir)) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint step directory could not be created");
    }
    if (!join_path(ranks_dir, sizeof(ranks_dir), step_dir, "ranks")) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint ranks path is too long");
    }
    if (!ensure_directory(ranks_dir)) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint ranks directory could not be created");
    }
    snprintf(rank_file_name, sizeof(rank_file_name), "rank_%06u.json", ctx->desc.global_rank);
    if (!join_path(rank_path, sizeof(rank_path), ranks_dir, rank_file_name)) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint rank shard path is too long");
    }
    status = write_rank_checkpoint(ctx, rank_path);
    if (status != CAIRN_OK) {
        return status;
    }

    if (!join_path(manifest_path, sizeof(manifest_path), step_dir, "manifest.json")) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint manifest path is too long");
    }
    if (snprintf(manifest_tmp, sizeof(manifest_tmp), "%s.tmp", manifest_path) <= 0 || strlen(manifest_path) + 4 >= sizeof(manifest_tmp)) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint temporary manifest path is too long");
    }
    file = fopen(manifest_tmp, "wb");
    if (file == NULL) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint manifest could not be opened for writing");
    }
    fprintf(file, "{\n");
    fprintf(file, "  \"version\": 1,\n");
    fprintf(file, "  \"plan_id\": \"%s\",\n", ctx->plan_id);
    fprintf(file, "  \"world_size\": %u,\n", ctx->plan_world_size);
    fprintf(file, "  \"step\": %llu,\n", (unsigned long long)ctx->step);
    fprintf(file, "  \"rank_count\": 1,\n");
    fprintf(file, "  \"rank_shard_path\": \"ranks/%s\",\n", rank_file_name);
    fprintf(file, "  \"complete\": true\n");
    fprintf(file, "}\n");
    if (fclose(file) != 0) {
        remove(manifest_tmp);
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint manifest could not be closed");
    }
    if (!atomic_publish(manifest_tmp, manifest_path)) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint manifest could not be published");
    }

    if (!join_path(latest_path, sizeof(latest_path), tag, "latest.json")) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint latest path is too long");
    }
    if (!join_path(manifest_rel, sizeof(manifest_rel), step_name, "manifest.json")) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint relative manifest path is too long");
    }
    if (snprintf(latest_tmp, sizeof(latest_tmp), "%s.tmp", latest_path) <= 0 || strlen(latest_path) + 4 >= sizeof(latest_tmp)) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint temporary latest path is too long");
    }
    file = fopen(latest_tmp, "wb");
    if (file == NULL) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint latest manifest could not be opened for writing");
    }
    fprintf(file, "{\n");
    fprintf(file, "  \"version\": 1,\n");
    fprintf(file, "  \"plan_id\": \"%s\",\n", ctx->plan_id);
    fprintf(file, "  \"step\": %llu,\n", (unsigned long long)ctx->step);
    fprintf(file, "  \"manifest_path\": \"%s\",\n", manifest_rel);
    fprintf(file, "  \"complete\": true\n");
    fprintf(file, "}\n");
    if (fclose(file) != 0) {
        remove(latest_tmp);
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint latest manifest could not be closed");
    }
    if (!atomic_publish(latest_tmp, latest_path)) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint latest manifest could not be published");
    }
    return CAIRN_OK;
}

int cairn_finalize(cairn_context_t *ctx) {
    if (ctx == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    ctx->finalized = 1;
    free(ctx->ops);
    ctx->ops = NULL;
    free(ctx->segments);
    ctx->segments = NULL;
    free(ctx->tensors);
    ctx->tensors = NULL;
    free(ctx->dependency_refs);
    ctx->dependency_refs = NULL;
    free(ctx->tensor_refs);
    ctx->tensor_refs = NULL;
    clear_dataset(ctx);
    free(ctx->trace_events);
    ctx->trace_events = NULL;
    free(ctx->scheduler_marks);
    ctx->scheduler_marks = NULL;
    free(ctx->op_logical_end);
    ctx->op_logical_end = NULL;
    free(ctx->arena);
    ctx->arena = NULL;
    free(ctx);
    return CAIRN_OK;
}

const char *cairn_last_error(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return "no context";
    }
    return ctx->last_error;
}

const char *cairn_plan_id(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return "";
    }
    return ctx->plan_id;
}

uint64_t cairn_plan_op_count(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->op_count;
}

uint64_t cairn_plan_memory_bytes(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->estimated_memory_bytes;
}

uint64_t cairn_memory_segment_count(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->segment_count;
}

uint64_t cairn_plan_tensor_count(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->tensor_count;
}

uint64_t cairn_plan_dependency_ref_count(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->dependency_ref_count;
}

uint64_t cairn_plan_tensor_ref_count(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->tensor_ref_count;
}

uint64_t cairn_memory_arena_bytes(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->arena_bytes;
}

int cairn_host_arena_allocated(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->arena_allocated;
}

uint64_t cairn_plan_input_token_count(const cairn_context_t *ctx) {
    return plan_input_token_count(ctx);
}

uint64_t cairn_dataset_shard_count(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->dataset_shard_count;
}

uint64_t cairn_dataset_total_tokens(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->dataset_total_tokens;
}

uint32_t cairn_dataset_token_bytes(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->dataset_token_bytes;
}

uint64_t cairn_current_step(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->step;
}
