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
    cairn_init_desc_t desc;
    unsigned int world_size;

    if (argc != 4) {
        fprintf(stderr, "usage: %s RANK_PLAN WORLD_SIZE CHECKPOINT_ROOT\n", argv[0]);
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
    desc.checkpoint_path = argv[3];
    desc.dataset_manifest_path = NULL;

    if (expect_ok(cairn_init(&ctx, &desc), "cairn_init", ctx)) {
        return 1;
    }
    if (expect_ok(cairn_load_plan(ctx, argv[1]), "cairn_load_plan", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_load_checkpoint(ctx, argv[3]), "cairn_load_checkpoint", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (cairn_current_step(ctx) == 0) {
        fprintf(stderr, "checkpoint restore returned step 0\n");
        cairn_finalize(ctx);
        return 1;
    }
    if (expect_ok(cairn_finalize(ctx), "cairn_finalize", ctx)) {
        return 1;
    }

    printf("cairn restore smoke ok\n");
    return 0;
}
