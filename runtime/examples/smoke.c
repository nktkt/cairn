#include "cairn/cairn.h"

#include <stdio.h>

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
    cairn_init_desc_t desc;

    if (argc != 2) {
        fprintf(stderr, "usage: %s PLAN_MANIFEST\n", argv[0]);
        return 2;
    }

    desc.global_rank = 0;
    desc.world_size = 1;
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
    if (expect_ok(cairn_next_batch(ctx, &batch), "cairn_next_batch", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_train_step(ctx, &batch), "cairn_train_step", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_save_checkpoint(ctx, "smoke"), "cairn_save_checkpoint", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_finalize(ctx), "cairn_finalize", ctx)) {
        return 1;
    }

    printf("cairn runtime smoke ok\n");
    return 0;
}
