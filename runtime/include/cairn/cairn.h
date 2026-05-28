#ifndef CAIRN_CAIRN_H
#define CAIRN_CAIRN_H

#include <stdint.h>

#define CAIRN_DATASET_PATH_MAX 1024

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
    uint64_t shard_token_offset;
    uint64_t tokens_available;
    uint32_t microbatch_size;
    uint32_t shard_index;
    char shard_path[CAIRN_DATASET_PATH_MAX];
} cairn_batch_t;

typedef enum {
    CAIRN_OK = 0,
    CAIRN_ERR_INVALID_ARGUMENT = 1,
    CAIRN_ERR_IO = 2,
    CAIRN_ERR_PLAN = 3,
    CAIRN_ERR_STATE = 4
} cairn_status_t;

typedef struct {
    uint64_t steps_executed;
    uint64_t ops_executed;
    uint64_t compute_ops;
    uint64_t communication_ops;
    uint64_t io_ops;
    uint64_t compute_bytes;
    uint64_t communication_bytes;
    uint64_t io_bytes;
} cairn_runtime_stats_t;

typedef struct {
    uint64_t step;
    uint64_t ordinal;
    uint64_t bytes;
    uint64_t logical_start;
    uint64_t logical_end;
    uint32_t op_id;
    uint32_t tick;
    uint32_t dep_count;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t op_class;
    char kind[48];
    char stream[32];
} cairn_trace_event_t;

int cairn_init(cairn_context_t **ctx, const cairn_init_desc_t *desc);
int cairn_load_plan(cairn_context_t *ctx, const char *path);
int cairn_load_dataset(cairn_context_t *ctx, const char *path);
int cairn_load_checkpoint(cairn_context_t *ctx, const char *path);
int cairn_next_batch(cairn_context_t *ctx, cairn_batch_t *batch);
int cairn_read_batch_tokens(
    cairn_context_t *ctx,
    const cairn_batch_t *batch,
    void *out,
    uint64_t token_count,
    uint64_t out_nbytes,
    uint64_t *out_tokens_read
);
int cairn_train_step(cairn_context_t *ctx, const cairn_batch_t *batch);
int cairn_get_stats(const cairn_context_t *ctx, cairn_runtime_stats_t *stats);
uint64_t cairn_trace_event_count(const cairn_context_t *ctx);
int cairn_get_trace_event(const cairn_context_t *ctx, uint64_t index, cairn_trace_event_t *event);
int cairn_write_trace(cairn_context_t *ctx, const char *path);
int cairn_should_checkpoint(cairn_context_t *ctx);
int cairn_save_checkpoint(cairn_context_t *ctx, const char *tag);
int cairn_finalize(cairn_context_t *ctx);
const char *cairn_last_error(const cairn_context_t *ctx);
const char *cairn_plan_id(const cairn_context_t *ctx);
uint64_t cairn_plan_op_count(const cairn_context_t *ctx);
uint64_t cairn_plan_memory_bytes(const cairn_context_t *ctx);
uint64_t cairn_memory_segment_count(const cairn_context_t *ctx);
uint64_t cairn_plan_tensor_count(const cairn_context_t *ctx);
uint64_t cairn_plan_dependency_ref_count(const cairn_context_t *ctx);
uint64_t cairn_plan_tensor_ref_count(const cairn_context_t *ctx);
uint64_t cairn_memory_arena_bytes(const cairn_context_t *ctx);
uint64_t cairn_dataset_shard_count(const cairn_context_t *ctx);
uint64_t cairn_dataset_total_tokens(const cairn_context_t *ctx);
uint32_t cairn_dataset_token_bytes(const cairn_context_t *ctx);
uint64_t cairn_current_step(const cairn_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif
