// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#include "dialect-mlp.h"
#include "ops-mlp.h"

using namespace mlir;
using namespace mlir::mlp;

void MlpDialect::initialize() {
    addOperations<
#define GET_OP_LIST 
#include "ops-mlp.cpp.inc"
    >();
}
