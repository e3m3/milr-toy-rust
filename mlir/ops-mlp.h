// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __OPS_MLP_H__
#define __OPS_MLP_H__

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "ops-mlp.h.inc"

#endif // __OPS_MLP_H__
