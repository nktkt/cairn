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

typedef struct {
    uint64_t op_id;
    uint64_t bytes;
    cairn_op_class_t op_class;
    char kind[48];
    char stream[32];
} cairn_plan_op_t;

typedef struct {
    uint64_t offset;
    uint64_t nbytes;
    char name[64];
} cairn_memory_segment_t;

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
    uint64_t segment_count;
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

static int restore_checkpoint_contents(cairn_context_t *ctx, const char *contents) {
    char checkpoint_plan_id[65];
    uint64_t parsed;

    if (ctx == NULL || contents == NULL) {
        return 0;
    }
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

    if (snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path) <= 0 || strlen(path) + 4 >= sizeof(tmp_path)) {
        return set_error(ctx, CAIRN_ERR_IO, "rank checkpoint temporary path is too long");
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
    fprintf(file, "  \"arena_bytes\": %llu,\n", (unsigned long long)ctx->arena_bytes);
    fprintf(file, "  \"arena_allocated\": %s,\n", ctx->arena_allocated ? "true" : "false");
    fprintf(file, "  \"ops_executed\": %llu,\n", (unsigned long long)ctx->stats.ops_executed);
    fprintf(file, "  \"compute_ops\": %llu,\n", (unsigned long long)ctx->stats.compute_ops);
    fprintf(file, "  \"communication_ops\": %llu,\n", (unsigned long long)ctx->stats.communication_ops);
    fprintf(file, "  \"io_ops\": %llu,\n", (unsigned long long)ctx->stats.io_ops);
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
    free(ctx->arena);
    ctx->arena = NULL;
    ctx->op_count = 0;
    ctx->segment_count = 0;
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
    created->segment_count = 0;
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
    ctx->plan_loaded = 1;
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

    batch->step = ctx->step;
    batch->token_offset = ctx->token_offset;
    batch->microbatch_size = ctx->plan_microbatch_size;
    return CAIRN_OK;
}

int cairn_train_step(cairn_context_t *ctx, const cairn_batch_t *batch) {
    uint64_t index;

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

    for (index = 0; index < ctx->op_count; index++) {
        const cairn_plan_op_t *op = &ctx->ops[index];

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
    ctx->stats.steps_executed += 1;
    ctx->step += 1;
    ctx->token_offset += batch->microbatch_size;
    return CAIRN_OK;
}

int cairn_get_stats(const cairn_context_t *ctx, cairn_runtime_stats_t *stats) {
    if (ctx == NULL || stats == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    *stats = ctx->stats;
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

uint64_t cairn_memory_arena_bytes(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->arena_bytes;
}

uint64_t cairn_current_step(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->step;
}
