// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

use std::collections::HashMap;
use std::fmt;
use std::hash;

use crate::ast;
use crate::exit_code;
use crate::options;

use ast::Ast;
use ast::AcceptGen;
use ast::AstGenerator;
use ast::Expr;
use ast::ExprKind;
use ast::ExprKindID;
use ast::GenResult;
use ast::Shape;
use ast::Symbol;
use ast::TMlp;
use ast::Type;
use ast::TypeMap;
use ast::BinopExpr;
use ast::CallExpr;
use ast::FunctionExpr;
use ast::LiteralExpr;
use ast::ModuleExpr;
use ast::NumberExpr;
use ast::PrintExpr;
use ast::PrototypeExpr;
use ast::ReturnExpr;
use ast::TransposeExpr;
use ast::VarExpr;
use ast::VarDeclExpr;
use exit_code::exit;
use exit_code::ExitCode;
use options::RunOptions;
use options::VerboseMode;

pub struct TypeCheck<'a> {
    options: &'a RunOptions,
    map: TypeMap,
}

impl <'a> TypeCheck<'a> {
    pub fn new(options: &'a RunOptions) -> Self {
        TypeCheck{options, map: TypeMap::new()}
    }

    fn check_binop(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        todo!()
    }

    fn check_call(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        todo!()
    }

    fn check_function(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        todo!()
    }

    fn check_literal(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        todo!()
    }

    fn check_module(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        let expr: &ModuleExpr = ast.as_impl().get_kind().to_module().unwrap();
        if self.options.is_verbose(VerboseMode::Sem) {
            eprintln!("Running TypeCheck on module '{}'", expr.get_name());
        }
        let mut type_map = Default::default();
        for function in expr.get_functions().iter() {
            match function.accept_gen(self, None) {
                Err(msg)    => return Err(msg),
                Ok(tm)      => type_map = tm,
            }
        }
        Ok(type_map)
    }

    fn check_number(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        todo!()
    }

    fn check_print(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        todo!()
    }

    fn check_prototype(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        todo!()
    }

    fn check_return(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        todo!()
    }

    fn check_transpose(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        todo!()
    }

    fn check_var(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        todo!()
    }

    fn check_var_decl(&mut self, ast: &dyn Ast<Expr>) -> GenResult<TypeMap> {
        todo!()
    }
}

impl <'a> AstGenerator<Expr, TypeMap> for TypeCheck<'a> {
    fn gen(&mut self, ast: &dyn Ast<Expr>, _acc: Option<TypeMap>) -> GenResult<TypeMap> {
        match ast.as_impl().get_kind_id() {
            ExprKindID::Binop       => self.check_binop(ast),
            ExprKindID::Call        => self.check_call(ast),
            ExprKindID::Function    => self.check_function(ast),
            ExprKindID::Literal     => self.check_literal(ast),
            ExprKindID::Module      => self.check_module(ast),
            ExprKindID::Number      => self.check_number(ast),
            ExprKindID::Print       => self.check_print(ast),
            ExprKindID::Prototype   => self.check_prototype(ast),
            ExprKindID::Return      => self.check_return(ast),
            ExprKindID::Transpose   => self.check_transpose(ast),
            ExprKindID::Var         => self.check_var(ast),
            ExprKindID::VarDecl     => self.check_var_decl(ast),
            ExprKindID::Unset       => {
                eprintln!("Unexpected AST of kind Unset {}", ast.get_loc());
                exit(ExitCode::SemanticError);
            },
        }
    }
}
