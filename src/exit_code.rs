// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

use std::process;

#[repr(u8)]
#[derive(Clone,Copy)]
pub enum ExitCode {
    Ok              = 0,
    BuildError      = 1,
    AstError        = 2,
    ArgParseError   = 3,
    LexerError      = 4,
    ParserError     = 5,
    SemanticError   = 6,
    ModuleError     = 7,
    IRGenError      = 8,
    MainGenError    = 9,
    MainGenCError   = 10,
    VerifyError     = 11,
    TargetError     = 12,
    LinkError       = 13,
    WriteError      = 14,
    CommandError    = 15,
}

pub fn exit(code: ExitCode) -> ! {
    process::exit(code as i32);
}
