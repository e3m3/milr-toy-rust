// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __DIALECT_MLP__
#define __DIALECT_MLP__

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def MlpDialect : Dialect {
    let name = "mlp";
    let summary = "A simple dialect for analysis and optimization of the MLP language";
    let description = [{
        A dialect implementing the MLP language for operations over floating point values
        and tensors.
    }];
    let cppNamespace = "::mlir::mlp";
}

class MlpOp<string mnemonic, list<Trait> traits = []> :
    Op<MlpDialect, mnemonic, traits>;

#endif // __DIALECT_MLP__
