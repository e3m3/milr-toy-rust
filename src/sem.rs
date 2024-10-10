// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

use crate::ast;
use crate::exit_code;
use crate::options;
use crate::sem_decl;
use crate::sem_type;

use ast::Ast;
use ast::AcceptGen;
use ast::AcceptVisit;
use ast::Expr;
use ast::TypeMap;
use exit_code::exit;
use exit_code::ExitCode;
use options::RunOptions;
use sem_decl::DeclCheck;
use sem_type::TypeCheck;

pub struct Semantics {}

impl Semantics {
    pub fn check_all(ast: &dyn Ast<Expr>, options: &RunOptions) -> TypeMap {
        let mut decl_check = DeclCheck::new(options);
        let decl_result: bool = ast.as_impl().accept(&mut decl_check);
        if !decl_result {
            eprintln!("AST failed DeclCheck semantics check");
            exit(ExitCode::SemanticError);
        }
        let mut type_check = TypeCheck::new(options);
        let type_map: TypeMap = match ast.as_impl().accept_gen(&mut type_check, None) {
            Ok(tm)      => tm,
            Err(msg)    => {
                eprintln!("AST failed TypeCheck semantics check: {}", msg);
                exit(ExitCode::SemanticError);
            },
        };
        if options.sem_exit { exit(ExitCode::Ok); }
        type_map
    }
}
