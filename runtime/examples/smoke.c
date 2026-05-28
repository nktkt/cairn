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

static unsigned int read_u32_le(const unsigned char *bytes) {
    return ((unsigned int)bytes[0])
         | ((unsigned int)bytes[1] << 8)
         | ((unsigned int)bytes[2] << 16)
         | ((unsigned int)bytes[3] << 24);
}

int main(int argc, char **argv) {
    cairn_context_t *ctx = NULL;
    cairn_batch_t batch;
    cairn_batch_t wrap_batch;
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
    unsigned char token_buffer[40];
    unsigned char staged_token[4];
    uint64_t tokens_read;
    uint64_t tokens_staged;
    uint64_t input_token_count;
    int status;

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
        if (expect_ok(
                cairn_read_batch_tokens(ctx, &batch, token_buffer, 10, sizeof(token_buffer), &tokens_read),
                "cairn_read_batch_tokens",
                ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (tokens_read != 10 || read_u32_le(token_buffer) != 0 || read_u32_le(token_buffer + 36) != 9) {
            fprintf(stderr, "dataset batch token bytes were not read correctly\n");
            cairn_finalize(ctx);
            return 1;
        }
        status = cairn_read_batch_tokens(ctx, &batch, token_buffer, 10, 4, &tokens_read);
        if (status == CAIRN_OK) {
            fprintf(stderr, "dataset batch token reader accepted an undersized buffer\n");
            cairn_finalize(ctx);
            return 1;
        }
        wrap_batch = batch;
        wrap_batch.token_offset = UINT64_MAX - 1;
        if (expect_ok(
                cairn_read_batch_tokens(ctx, &wrap_batch, token_buffer, 4, sizeof(token_buffer), &tokens_read),
                "cairn_read_batch_tokens_wrap",
                ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (tokens_read != 4 || read_u32_le(token_buffer) != 14 || read_u32_le(token_buffer + 12) != 1) {
            fprintf(stderr, "dataset batch token reader did not wrap a large logical offset correctly\n");
            cairn_finalize(ctx);
            return 1;
        }
        if (cairn_host_arena_allocated(ctx)) {
            input_token_count = cairn_plan_input_token_count(ctx);
            if (input_token_count == 0) {
                fprintf(stderr, "plan input token count was not populated\n");
                cairn_finalize(ctx);
                return 1;
            }
            if (expect_ok(cairn_stage_batch_input(ctx, &batch, &tokens_staged), "cairn_stage_batch_input", ctx)) {
                cairn_finalize(ctx);
                return 1;
            }
            if (tokens_staged != input_token_count) {
                fprintf(stderr, "staged input token count does not match plan input token count\n");
                cairn_finalize(ctx);
                return 1;
            }
            if (expect_ok(cairn_copy_tensor_bytes(ctx, "input_tokens", 0, staged_token, sizeof(staged_token)),
                          "cairn_copy_tensor_bytes_first",
                          ctx)) {
                cairn_finalize(ctx);
                return 1;
            }
            if (read_u32_le(staged_token) != 0) {
                fprintf(stderr, "staged input tensor did not start with the batch cursor token\n");
                cairn_finalize(ctx);
                return 1;
            }
            if (expect_ok(
                    cairn_copy_tensor_bytes(
                        ctx,
                        "input_tokens",
                        (tokens_staged - 1) * sizeof(staged_token),
                        staged_token,
                        sizeof(staged_token)),
                    "cairn_copy_tensor_bytes_last",
                    ctx)) {
                cairn_finalize(ctx);
                return 1;
            }
            if (read_u32_le(staged_token) != 15) {
                fprintf(stderr, "staged input tensor did not wrap through the dataset shards\n");
                cairn_finalize(ctx);
                return 1;
            }
            if (expect_ok(cairn_stage_batch_input(ctx, &wrap_batch, &tokens_staged),
                          "cairn_stage_batch_input_wrap",
                          ctx)) {
                cairn_finalize(ctx);
                return 1;
            }
            if (expect_ok(cairn_copy_tensor_bytes(ctx, "input_tokens", 0, staged_token, sizeof(staged_token)),
                          "cairn_copy_tensor_bytes_wrap_first",
                          ctx)) {
                cairn_finalize(ctx);
                return 1;
            }
            if (read_u32_le(staged_token) != 14) {
                fprintf(stderr, "explicit staging did not overwrite input tensor with the requested batch cursor\n");
                cairn_finalize(ctx);
                return 1;
            }
        }
    }
    if (expect_ok(cairn_train_step(ctx, &batch), "cairn_train_step", ctx)) {
        cairn_finalize(ctx);
        return 1;
    }
    if (argc == 6 && cairn_host_arena_allocated(ctx)) {
        if (expect_ok(cairn_copy_tensor_bytes(ctx, "input_tokens", 0, staged_token, sizeof(staged_token)),
                      "cairn_copy_tensor_bytes_after_train",
                      ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (read_u32_le(staged_token) != 0) {
            fprintf(stderr, "train step did not stage the current batch into input tensor\n");
            cairn_finalize(ctx);
            return 1;
        }
        if (expect_ok(cairn_copy_tensor_bytes(ctx, "activation_slot_0", 0, staged_token, sizeof(staged_token)),
                      "cairn_copy_tensor_bytes_activation",
                      ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (read_u32_le(staged_token) == 0) {
            fprintf(stderr, "host tensor executor did not mutate activation tensor\n");
            cairn_finalize(ctx);
            return 1;
        }
        if (expect_ok(cairn_copy_tensor_bytes(ctx, "gradients_shard", 0, staged_token, sizeof(staged_token)),
                      "cairn_copy_tensor_bytes_gradients",
                      ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (read_u32_le(staged_token) == 0) {
            fprintf(stderr, "host tensor executor did not mutate gradient tensor\n");
            cairn_finalize(ctx);
            return 1;
        }
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
    if (argc == 6) {
        if (expect_ok(cairn_next_batch(ctx, &batch), "cairn_next_batch_restore", ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (expect_ok(
                cairn_read_batch_tokens(ctx, &batch, token_buffer, 4, sizeof(token_buffer), &tokens_read),
                "cairn_read_batch_tokens_restore",
                ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (tokens_read != 4 || read_u32_le(token_buffer) != 0 || read_u32_le(token_buffer + 12) != 3) {
            fprintf(stderr, "checkpoint restore did not recover readable dataset token cursor\n");
            cairn_finalize(ctx);
            return 1;
        }
    }
    if (cairn_host_arena_allocated(ctx)) {
        if (expect_ok(cairn_copy_tensor_bytes(ctx, "activation_slot_0", 0, staged_token, sizeof(staged_token)),
                      "cairn_copy_tensor_bytes_restore_activation",
                      ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (read_u32_le(staged_token) == 0) {
            fprintf(stderr, "checkpoint restore did not recover activation tensor snapshot\n");
            cairn_finalize(ctx);
            return 1;
        }
        if (expect_ok(cairn_copy_tensor_bytes(ctx, "gradients_shard", 0, staged_token, sizeof(staged_token)),
                      "cairn_copy_tensor_bytes_restore_gradients",
                      ctx)) {
            cairn_finalize(ctx);
            return 1;
        }
        if (read_u32_le(staged_token) == 0) {
            fprintf(stderr, "checkpoint restore did not recover gradient tensor snapshot\n");
            cairn_finalize(ctx);
            return 1;
        }
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
