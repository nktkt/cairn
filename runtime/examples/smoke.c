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
    cairn_init_desc_t desc;
    unsigned int world_size;
    char plan_id[65];
    unsigned long long op_count;
    unsigned long long memory_bytes;

    if (argc != 4) {
        fprintf(stderr, "usage: %s RANK_PLAN WORLD_SIZE CHECKPOINT_OUT\n", argv[0]);
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
    desc.dataset_manifest_path = NULL;

    if (expect_ok(cairn_init(&ctx, &desc), "cairn_init", ctx)) {
        return 1;
    }
    if (expect_ok(cairn_load_plan(ctx, argv[1]), "cairn_load_plan", ctx)) {
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
    if (expect_ok(cairn_next_batch(ctx, &batch), "cairn_next_batch", ctx)) {
        cairn_finalize(ctx);
        return 1;
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
    snprintf(plan_id, sizeof(plan_id), "%s", cairn_plan_id(ctx));
    op_count = (unsigned long long)cairn_plan_op_count(ctx);
    memory_bytes = (unsigned long long)cairn_plan_memory_bytes(ctx);
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

    printf("cairn runtime smoke ok plan=%s ops=%llu memory=%llu\n",
           plan_id,
           op_count,
           memory_bytes);
    return 0;
}
