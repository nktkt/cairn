#include "cairn/cairn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

static uint64_t count_occurrences(const char *text, const char *needle) {
    uint64_t count;
    size_t needle_len;
    const char *cursor;

    if (text == NULL || needle == NULL || needle[0] == '\0') {
        return 0;
    }
    count = 0;
    needle_len = strlen(needle);
    cursor = text;
    while ((cursor = strstr(cursor, needle)) != NULL) {
        count++;
        cursor += needle_len;
    }
    return count;
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
    if (!parse_json_string(contents, "plan_id", ctx->plan_id, sizeof(ctx->plan_id))) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_PLAN, "plan file does not contain plan_id");
    }
    if (parse_json_u64(contents, "world_size", &parsed)) {
        if (parsed == 0 || parsed > UINT32_MAX) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "plan world_size is invalid");
        }
        ctx->plan_world_size = (uint32_t)parsed;
        if (ctx->desc.world_size != 0 && ctx->desc.world_size != ctx->plan_world_size) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "runtime world_size does not match plan world_size");
        }
    } else {
        ctx->plan_world_size = ctx->desc.world_size;
    }
    if (parse_json_u64(contents, "global_rank", &parsed)) {
        if (parsed > UINT32_MAX) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "plan global_rank is invalid");
        }
        ctx->plan_rank = (uint32_t)parsed;
        if (ctx->plan_rank != ctx->desc.global_rank) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "runtime global_rank does not match rank plan");
        }
    } else {
        ctx->plan_rank = ctx->desc.global_rank;
    }
    if (parse_json_u64(contents, "microbatch_size", &parsed)) {
        if (parsed == 0 || parsed > UINT32_MAX) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "plan microbatch_size is invalid");
        }
        ctx->plan_microbatch_size = (uint32_t)parsed;
    } else {
        ctx->plan_microbatch_size = 1;
    }
    if (parse_json_u64(contents, "estimated_bytes_per_rank", &parsed)) {
        ctx->estimated_memory_bytes = parsed;
    } else {
        ctx->estimated_memory_bytes = 0;
    }
    ctx->op_count = count_occurrences(contents, "\"op_id\"");
    free(contents);

    ctx->plan_loaded = 1;
    return CAIRN_OK;
}

int cairn_load_checkpoint(cairn_context_t *ctx, const char *path) {
    char *contents;
    size_t contents_size;
    char checkpoint_plan_id[65];
    uint64_t parsed;

    if (ctx == NULL || path == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!file_exists(path)) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint file does not exist");
    }

    contents = read_file(path, &contents_size);
    if (contents == NULL || contents_size == 0) {
        free(contents);
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint file could not be read");
    }
    if (parse_json_string(contents, "plan_id", checkpoint_plan_id, sizeof(checkpoint_plan_id))) {
        if (ctx->plan_loaded && strcmp(checkpoint_plan_id, ctx->plan_id) != 0) {
            free(contents);
            return set_error(ctx, CAIRN_ERR_PLAN, "checkpoint plan_id does not match loaded plan");
        }
    }
    if (parse_json_u64(contents, "step", &parsed)) {
        ctx->step = parsed;
    }
    if (parse_json_u64(contents, "token_offset", &parsed)) {
        ctx->token_offset = parsed;
    }
    free(contents);
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

    ctx->step += 1;
    ctx->token_offset += batch->microbatch_size;
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

    if (ctx == NULL || tag == NULL || tag[0] == '\0') {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!ctx->plan_loaded) {
        return set_error(ctx, CAIRN_ERR_STATE, "plan must be loaded before checkpointing");
    }
    file = fopen(tag, "wb");
    if (file == NULL) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint file could not be opened for writing");
    }
    fprintf(file, "{\n");
    fprintf(file, "  \"version\": 1,\n");
    fprintf(file, "  \"plan_id\": \"%s\",\n", ctx->plan_id);
    fprintf(file, "  \"global_rank\": %u,\n", ctx->desc.global_rank);
    fprintf(file, "  \"world_size\": %u,\n", ctx->plan_world_size);
    fprintf(file, "  \"step\": %llu,\n", (unsigned long long)ctx->step);
    fprintf(file, "  \"token_offset\": %llu,\n", (unsigned long long)ctx->token_offset);
    fprintf(file, "  \"op_count\": %llu,\n", (unsigned long long)ctx->op_count);
    fprintf(file, "  \"estimated_memory_bytes\": %llu\n", (unsigned long long)ctx->estimated_memory_bytes);
    fprintf(file, "}\n");
    if (fclose(file) != 0) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint file could not be closed");
    }
    return CAIRN_OK;
}

int cairn_finalize(cairn_context_t *ctx) {
    if (ctx == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    ctx->finalized = 1;
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

uint64_t cairn_current_step(const cairn_context_t *ctx) {
    if (ctx == NULL) {
        return 0;
    }
    return ctx->step;
}
