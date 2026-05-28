#include "cairn/cairn.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct cairn_context {
    cairn_init_desc_t desc;
    char last_error[256];
    uint64_t step;
    uint64_t token_offset;
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

static int file_contains(const char *path, const char *needle) {
    FILE *file;
    char buffer[4096];
    size_t needle_len;

    if (path == NULL || needle == NULL) {
        return 0;
    }

    needle_len = strlen(needle);
    file = fopen(path, "rb");
    if (file == NULL) {
        return 0;
    }

    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        if (strstr(buffer, needle) != NULL || needle_len == 0) {
            fclose(file);
            return 1;
        }
    }

    fclose(file);
    return 0;
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
    created->step = 0;
    created->token_offset = 0;
    created->plan_loaded = 0;
    created->checkpoint_loaded = 0;
    created->finalized = 0;
    created->last_error[0] = '\0';
    *ctx = created;
    return CAIRN_OK;
}

int cairn_load_plan(cairn_context_t *ctx, const char *path) {
    if (ctx == NULL || path == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!file_exists(path)) {
        return set_error(ctx, CAIRN_ERR_IO, "plan file does not exist");
    }
    if (!file_contains(path, "\"plan_id\"")) {
        return set_error(ctx, CAIRN_ERR_PLAN, "plan file does not contain plan_id");
    }
    ctx->plan_loaded = 1;
    return CAIRN_OK;
}

int cairn_load_checkpoint(cairn_context_t *ctx, const char *path) {
    if (ctx == NULL || path == NULL) {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!file_exists(path)) {
        return set_error(ctx, CAIRN_ERR_IO, "checkpoint file does not exist");
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
    batch->microbatch_size = 1;
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
    if (ctx == NULL || tag == NULL || tag[0] == '\0') {
        return CAIRN_ERR_INVALID_ARGUMENT;
    }
    if (ctx->finalized) {
        return set_error(ctx, CAIRN_ERR_STATE, "context is finalized");
    }
    if (!ctx->plan_loaded) {
        return set_error(ctx, CAIRN_ERR_STATE, "plan must be loaded before checkpointing");
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
