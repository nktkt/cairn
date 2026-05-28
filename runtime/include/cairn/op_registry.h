/* Generated from registry/op_registry.json. Do not edit by hand. */
#ifndef CAIRN_OP_REGISTRY_H
#define CAIRN_OP_REGISTRY_H

#include <stddef.h>

#define CAIRN_OP_REGISTRY_VERSION 1
#define CAIRN_OP_REGISTRY_SHA256 "8fbcf67fae9818f861e337b822663cd501a6357c0fa00b854c6f9c1878241511"

typedef enum {
    CAIRN_OP_CLASS_COMPUTE = 0,
    CAIRN_OP_CLASS_COMMUNICATION = 1,
    CAIRN_OP_CLASS_IO = 2
} cairn_op_class_t;

typedef struct {
    const char *kind;
    cairn_op_class_t op_class;
} cairn_executor_desc_t;

static const cairn_executor_desc_t CAIRN_EXECUTORS[] = {
    {"rmsnorm", CAIRN_OP_CLASS_COMPUTE},
    {"attention_fwd", CAIRN_OP_CLASS_COMPUTE},
    {"attention_bwd", CAIRN_OP_CLASS_COMPUTE},
    {"mlp_fwd", CAIRN_OP_CLASS_COMPUTE},
    {"mlp_bwd", CAIRN_OP_CLASS_COMPUTE},
    {"optimizer", CAIRN_OP_CLASS_COMPUTE},
    {"all_reduce", CAIRN_OP_CLASS_COMMUNICATION},
    {"reduce_scatter", CAIRN_OP_CLASS_COMMUNICATION},
    {"all_gather", CAIRN_OP_CLASS_COMMUNICATION},
    {"pipe_send_activation", CAIRN_OP_CLASS_COMMUNICATION},
    {"pipe_recv_activation_grad", CAIRN_OP_CLASS_COMMUNICATION},
    {"checkpoint_stage", CAIRN_OP_CLASS_IO}
};

static const size_t CAIRN_EXECUTOR_COUNT = sizeof(CAIRN_EXECUTORS) / sizeof(CAIRN_EXECUTORS[0]);

#endif
