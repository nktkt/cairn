#include "cairn/cairn.h"

#include <stdio.h>
#include <string.h>

static int expect_ok(int status, const char *label, cairn_context_t *ctx) {
    if (status != CAIRN_OK) {
        fprintf(stderr, "%s failed: %s\n", label, cairn_last_error(ctx));
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    cairn_context_t *ctx = NULL;
    cairn_batch_t batch;
    cairn_runtime_stats_t stats;
    cairn_trace_event_t first_trace;
    cairn_trace_event_t last_trace;
    cairn_init_desc_t desc;
    unsigned int world_size;
    char plan_id[65];
    unsigned long long op_count;
    unsigned long long memory_bytes;
    unsigned long long segment_count;
    unsigned long long tensor_count;
    unsigned long long dependency_count;
    unsigned long long tensor_ref_count;
    unsigned long long dataset_shard_count;
    unsigned long long dataset_total_tokens;
    unsigned long long arena_bytes;

    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage: %s RANK_PLAN WORLD_SIZE CHECKPOINT_OUT [TRACE_OUT] [DATASET_MANIFEST]\n", argv[0]);
        return 2;
    }
    if (sscanf(argv[2], "%u", &world_size) != 1 || world_size == 0) {
        fprintf(stderr, "WORLD_SIZE must be a positive integer\n");
        return 2;
    }

    desc.global_rank = 0;
    desc.world_size = world_size;
    desc.local_rank = 0;
    desc.plan_path = argv[1];
    desc.topology_path = NULL;
    desc.checkpoint_path = NULL;
    desc.dataset_manifest_path = argc == 6 ? argv[5] : NULL;

    if (expect_ok(cairn_init(&ctx, &desc), "cairn_init", ctx)) {
        return 1;
    }
    if (expect_ok(cairn_load_plan(ctx, argv[1]), "cairn_load_plan", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (argc == 6 && expect_ok(cairn_load_dataset(ctx, argv[5]), "cairn_load_dataset", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (cairn_plan_op_count(ctx) == 0) {
        fprintf(stderr, "rank plan has no ops\n");
        cairn_finalize(ctx);
        return 1;
    }
    if (cairn_plan_memory_bytes(ctx) == 0) {
        fprintf(stderr, "rank plan has no memory estimate\n");
        cairn_finalize(ctx);
        return 1;
    }
    if (cairn_memory_segment_count(ctx) == 0 || cairn_memory_arena_bytes(ctx) == 0) {
        fprintf(stderr, "rank plan has no memory arena metadata\n");
        cairn_finalize(ctx);
        return 1;
    }
    if (strstr(argv[1], ".cairn") != NULL
        && (cairn_plan_tensor_count(ctx) == 0 || cairn_plan_dependency_ref_count(ctx) == 0 || cairn_plan_tensor_ref_count(ctx) == 0)) {
        fprintf(stderr, "binary rank plan has no tensor or dependency metadata\n");
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_next_batch(ctx, &batch), "cairn_next_batch", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (argc == 6) {
        if (cairn_dataset_shard_count(ctx) == 0 || cairn_dataset_total_tokens(ctx) == 0 || cairn_dataset_token_bytes(ctx) == 0) {
            fprintf(stderr, "dataset manifest was not loaded\n");
            cairn_finalize(ctx);
            return 1;
        }
        if (batch.shard_path[0] == '\0' || batch.tokens_available == 0) {
            fprintf(stderr, "dataset batch cursor was not populated\n");
            cairn_finalize(ctx);
            return 1;
        }
    }
    if (expect_ok(cairn_train_step(ctx, &batch), "cairn_train_step", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_get_stats(ctx, &stats), "cairn_get_stats", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (stats.steps_executed != 1 || stats.ops_executed != cairn_plan_op_count(ctx)) {
        fprintf(stderr, "runtime stats did not track executed op table\n");
        cairn_finalize(ctx);
        return 1;
    }
    if (cairn_trace_event_count(ctx) != cairn_plan_op_count(ctx)) {
        fprintf(stderr, "runtime trace did not record executed op table\n");
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_get_trace_event(ctx, 0, &first_trace), "cairn_get_trace_event_first", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_get_trace_event(ctx, cairn_trace_event_count(ctx) - 1, &last_trace), "cairn_get_trace_event_last", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (first_trace.ordinal != 0 || first_trace.dep_count != 0 || last_trace.ordinal + 1 != cairn_trace_event_count(ctx)) {
        fprintf(stderr, "runtime trace has unexpected execution order metadata\n");
        cairn_finalize(ctx);
        return 1;
    }
    if (stats.compute_ops == 0 || stats.communication_ops == 0 || stats.io_ops == 0) {
        fprintf(stderr, "runtime stats did not classify compute/communication/io ops\n");
        cairn_finalize(ctx);
        return 1;
    }
    if (cairn_current_step(ctx) != 1) {
        fprintf(stderr, "unexpected step after train_step\n");
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_save_checkpoint(ctx, argv[3]), "cairn_save_checkpoint", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (argc >= 5 && expect_ok(cairn_write_trace(ctx, argv[4]), "cairn_write_trace", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    snprintf(plan_id, sizeof(plan_id), "%s", cairn_plan_id(ctx));
    op_count = (unsigned long long)cairn_plan_op_count(ctx);
    memory_bytes = (unsigned long long)cairn_plan_memory_bytes(ctx);
    segment_count = (unsigned long long)cairn_memory_segment_count(ctx);
    tensor_count = (unsigned long long)cairn_plan_tensor_count(ctx);
    dependency_count = (unsigned long long)cairn_plan_dependency_ref_count(ctx);
    tensor_ref_count = (unsigned long long)cairn_plan_tensor_ref_count(ctx);
    dataset_shard_count = (unsigned long long)cairn_dataset_shard_count(ctx);
    dataset_total_tokens = (unsigned long long)cairn_dataset_total_tokens(ctx);
    arena_bytes = (unsigned long long)cairn_memory_arena_bytes(ctx);
    if (expect_ok(cairn_finalize(ctx), "cairn_finalize", ctx)) {
        return 1;
    }
    ctx = NULL;
    if (expect_ok(cairn_init(&ctx, &desc), "cairn_init_restore", ctx)) {
        return 1;
    }
    if (expect_ok(cairn_load_plan(ctx, argv[1]), "cairn_load_plan_restore", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_load_checkpoint(ctx, argv[3]), "cairn_load_checkpoint", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (cairn_current_step(ctx) != 1) {
        fprintf(stderr, "checkpoint restore did not recover step\n");
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_finalize(ctx), "cairn_finalize_restore", ctx)) {
        return 1;
    }

    printf("cairn runtime smoke ok plan=%s ops=%llu memory=%llu segments=%llu tensors=%llu deps=%llu tensor_refs=%llu dataset_shards=%llu dataset_tokens=%llu arena=%llu\n",
           plan_id,
           op_count,
           memory_bytes,
           segment_count,
           tensor_count,
           dependency_count,
           tensor_ref_count,
           dataset_shard_count,
           dataset_total_tokens,
           arena_bytes);
    return 0;
}
