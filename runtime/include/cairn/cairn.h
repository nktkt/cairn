#ifndef CAIRN_CAIRN_H
#define CAIRN_CAIRN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cairn_context cairn_context_t;

typedef struct {
    uint32_t global_rank;
    uint32_t world_size;
    uint32_t local_rank;
    const char *plan_path;
    const char *topology_path;
    const char *checkpoint_path;
    const char *dataset_manifest_path;
} cairn_init_desc_t;

typedef struct {
    uint64_t step;
    uint64_t token_offset;
    uint32_t microbatch_size;
} cairn_batch_t;

typedef enum {
    CAIRN_OK = 0,
    CAIRN_ERR_INVALID_ARGUMENT = 1,
    CAIRN_ERR_IO = 2,
    CAIRN_ERR_PLAN = 3,
    CAIRN_ERR_STATE = 4
} cairn_status_t;

int cairn_init(cairn_context_t **ctx, const cairn_init_desc_t *desc);
int cairn_load_plan(cairn_context_t *ctx, const char *path);
int cairn_load_checkpoint(cairn_context_t *ctx, const char *path);
int cairn_next_batch(cairn_context_t *ctx, cairn_batch_t *batch);
int cairn_train_step(cairn_context_t *ctx, const cairn_batch_t *batch);
int cairn_should_checkpoint(cairn_context_t *ctx);
int cairn_save_checkpoint(cairn_context_t *ctx, const char *tag);
int cairn_finalize(cairn_context_t *ctx);
const char *cairn_last_error(const cairn_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif
