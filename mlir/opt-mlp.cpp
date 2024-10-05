// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "dialect-mlp.h"
#include "ops-mlp.cpp.inc"

int main(int argc, char **argv) {
    mlir::registerAllPasses();

    mlir::DialectRegistry registry{};
    registry.insert<mlir::mlp::MlpDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::affine::AffineDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();
    registry.insert<mlir::vector::VectorDialect>();
    registry.insert<mlir::gpu::GPUDialect>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "MLP Dialect optimization driver\n", registry)
    );
}
