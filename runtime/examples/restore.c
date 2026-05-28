#include "cairn/cairn.h"

#include <stdio.h>

static int expect_ok(int status, const char *label, cairn_context_t *ctx) {
    if (status != CAIRN_OK) {
        fprintf(stderr, "%s failed: %s\n", label, cairn_last_error(ctx));
        return 1;
    }
    return 0;
}

static unsigned int read_u32_le(const unsigned char *bytes) {
    return ((unsigned int)bytes[0])
         | ((unsigned int)bytes[1] << 8)
         | ((unsigned int)bytes[2] << 16)
         | ((unsigned int)bytes[3] << 24);
}

int main(int argc, char **argv) {
    cairn_context_t *ctx = NULL;
    cairn_init_desc_t desc;
    cairn_batch_t batch;
    unsigned char token_buffer[16];
    uint64_t tokens_read;
    unsigned int world_size;

    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s RANK_PLAN WORLD_SIZE CHECKPOINT_ROOT [DATASET_MANIFEST]\n", argv[0]);
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
    desc.dataset_manifest_path = argc == 5 ? argv[4] : NULL;

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
    if (argc == 5) {
        if (expect_ok(cairn_next_batch(ctx, &batch), "cairn_next_batch", ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (batch.shard_path[0] == '\0' || batch.shard_index != 0 || batch.shard_token_offset != 0) {
            fprintf(stderr, "checkpoint restore did not recover dataset cursor\n");
            cairn_finalize(ctx);
            return 1;
        }
        if (expect_ok(
                cairn_read_batch_tokens(ctx, &batch, token_buffer, 4, sizeof(token_buffer), &tokens_read),
                "cairn_read_batch_tokens",
                ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (tokens_read != 4 || read_u32_le(token_buffer) != 0 || read_u32_le(token_buffer + 12) != 3) {
            fprintf(stderr, "checkpoint restore did not recover readable dataset token bytes\n");
            cairn_finalize(ctx);
            return 1;
        }
    }
    if (cairn_host_arena_allocated(ctx) && cairn_plan_tensor_count(ctx) > 0) {
        if (expect_ok(cairn_copy_tensor_bytes(ctx, "input_tokens", 0, token_buffer, 4),
                      "cairn_copy_tensor_bytes_input_tokens",
                      ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (read_u32_le(token_buffer) != 0) {
            fprintf(stderr, "checkpoint restore did not recover input token tensor bytes\n");
            cairn_finalize(ctx);
            return 1;
        }
        if (expect_ok(cairn_copy_tensor_bytes(ctx, "activation_slot_0", 0, token_buffer, 4),
                      "cairn_copy_tensor_bytes_activation",
                      ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (read_u32_le(token_buffer) == 0) {
            fprintf(stderr, "checkpoint restore did not recover activation tensor bytes\n");
            cairn_finalize(ctx);
            return 1;
        }
        if (expect_ok(cairn_copy_tensor_bytes(ctx, "gradients_shard", 0, token_buffer, 4),
                      "cairn_copy_tensor_bytes_gradients",
                      ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (read_u32_le(token_buffer) == 0) {
            fprintf(stderr, "checkpoint restore did not recover gradient tensor bytes\n");
            cairn_finalize(ctx);
            return 1;
        }
    }
    if (expect_ok(cairn_finalize(ctx), "cairn_finalize", ctx)) {
        return 1;
    }

    printf("cairn restore smoke ok\n");
    return 0;
}
