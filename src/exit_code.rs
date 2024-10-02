// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

use std::process;

#[repr(u8)]
#[derive(Clone,Copy)]
pub enum ExitCode {
    Ok              = 0,
    AstError        = 1,
    ArgParseError   = 2,
    LexerError      = 3,
    ParserError     = 4,
    SemanticError   = 5,
    ModuleError     = 6,
    IRGenError      = 7,
    MainGenError    = 8,
    MainGenCError   = 9,
    VerifyError     = 10,
    TargetError     = 11,
    LinkError       = 12,
    WriteError      = 13,
    CommandError    = 14,
}

pub fn exit(code: ExitCode) -> ! {
    process::exit(code as i32);
}
