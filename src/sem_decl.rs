// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

use std::collections::HashSet;
use std::fmt;
use std::hash;

use crate::ast;
use crate::exit_code;
use crate::options;

use ast::Ast;
use ast::AcceptVisit;
use ast::AstVisitor;
use ast::Expr;
use ast::ExprKind;
use ast::ExprKindID;
use ast::Symbol;
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

struct Scope<T> {
    vars: HashSet<T>,
}

pub struct DeclCheck<'a> {
    options: &'a RunOptions,
    scope: Scope<Symbol>,
}

impl <T: Clone + fmt::Display + Eq + hash::Hash> Scope<T> {
    pub fn new() -> Self {
        Scope{vars: Default::default()}
    }

    pub fn iter(&self) -> impl Iterator + '_ {
        self.vars.iter()
    }

    /// Returns true if symbol is inserted in scope; else false
    pub fn add_symbol(&mut self, var: &T, options: &RunOptions) -> bool {
        let result = self.vars.insert(var.clone());
        if options.is_verbose(VerboseMode::Sem) && result {
            eprintln!("Added symbol '{}' to scope", *var);
        }
        result
    }

    /// Returns true if symbol is in scope; else false
    pub fn contains_symbol(&self, var: &T, options: &RunOptions) -> bool {
        let result = self.vars.contains(var);
        if options.is_verbose(VerboseMode::Sem) && result {
            eprintln!("Found symbol '{}' in scope", var);
        } else if options.is_verbose(VerboseMode::Sem) && !result {
            eprintln!("Found unbound symbol '{}' in scope", var);
        }
        result
    }

    /// Returns true if symbol is found and removed; else false
    pub fn remove_symbol(&mut self, var: &T, options: &RunOptions) -> bool {
        let result = self.vars.remove(var);
        if options.is_verbose(VerboseMode::Sem) && result {
            eprintln!("Removed symbol '{}' from scope", var);
        }
        result
    }
}

impl <T: Clone + fmt::Display + Eq + hash::Hash> Default for Scope<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl <'a> DeclCheck<'a> {
    pub fn new(options: &'a RunOptions) -> Self {
        let mut decl_check = DeclCheck{options, scope: Default::default()};
        // Add builtin symbols
        let expr_print = PrintExpr::new(Expr::default().as_value());
        let expr_transpose = TransposeExpr::new(Expr::default().as_value());
        assert!(decl_check.scope.add_symbol(&expr_print.get_symbol(), decl_check.options));
        assert!(decl_check.scope.add_symbol(&expr_transpose.get_symbol(), decl_check.options));
        decl_check
    }

    fn check_binop(&mut self, ast: &dyn Ast<Expr>) -> bool {
        let expr: &BinopExpr = ast.as_impl().get_kind().to_binop().unwrap();
        expr.get_lhs().accept(self) && expr.get_rhs().accept(self)
    }

    fn check_call(&mut self, ast: &dyn Ast<Expr>) -> bool {
        let expr: &CallExpr = ast.as_impl().get_kind().to_call().unwrap();
        let sym = ast.get_symbol().unwrap();
        if !self.scope.contains_symbol(&sym, self.options) {
            eprintln!("Expected to find function '{}' in scope {}'", sym, ast.get_loc());
            return false;
        }
        for arg in expr.get_args().iter() {
            if !arg.accept(self) {
                return false;
            }
        }
        true
    }

    // TODO: Forward declarations
    fn check_function(&mut self, ast: &dyn Ast<Expr>) -> bool {
        let expr: &FunctionExpr = ast.as_impl().get_kind().to_function().unwrap();
        let proto = expr.get_prototype();
        if !proto.accept(self) {
            return false;
        }
        let mut added: Vec<String> = Vec::new();
        for arg in proto.get_kind().to_prototype().unwrap().get_args().iter() {
            let sym = arg.get_symbol().unwrap();
            if !self.scope.add_symbol(&sym, self.options) {
                return false;
            }
            added.push(sym);
        }
        for value in expr.get_body().iter() {
            if !value.accept(self) {
                return false;
            }
            if value.isa(ExprKindID::VarDecl) {
                let sym = value.get_symbol().unwrap();
                added.push(sym.clone());
            }
        }
        for value in added.iter() {
            if !self.scope.remove_symbol(value, self.options) {
                eprintln!("Expected to find symbol '{}' in scope", *value);
                return false;
            }
        }
        let sym = proto.get_symbol().unwrap();
        if !self.scope.add_symbol(&sym, self.options) {
            eprintln!("Redeclaration of function '{}' {}", sym, ast.get_loc());
            false
        } else {
            true
        }
    }

    fn check_literal(&mut self, ast: &dyn Ast<Expr>) -> bool {
        true
    }

    fn check_module(&mut self, ast: &dyn Ast<Expr>) -> bool {
        let expr: &ModuleExpr = ast.as_impl().get_kind().to_module().unwrap();
        if self.options.is_verbose(VerboseMode::Sem) {
            eprintln!("Running DeclCheck on module '{}'", expr.get_name());
        }
        let mut added: Vec<String> = Vec::new();
        for function in expr.get_functions().iter() {
            let sym = function.get_symbol().unwrap();
            if self.scope.contains_symbol(&sym, self.options) {
                eprintln!("Redeclaration of function '{}' {}", sym, function.get_loc());
                return false;
            } else {
                added.push(sym);
            }
            if !function.accept(self) {
                return false;
            }
        }
        for function in added.iter() {
            let result = self.scope.remove_symbol(function, self.options);
            if !result {
                eprintln!("Expected to find function '{}' in scope", *function);
                return false;
            }
        }
        true
    }

    fn check_number(&mut self, ast: &dyn Ast<Expr>) -> bool {
        true
    }

    fn check_print(&mut self, ast: &dyn Ast<Expr>) -> bool {
        let expr: &PrintExpr = ast.as_impl().get_kind().to_print().unwrap();
        let sym = ast.get_symbol().unwrap();
        if !self.scope.contains_symbol(&sym, self.options) {
            eprintln!("Expected to find symbol '{}' in scope {}", sym, ast.get_loc());
            return false;
        }
        expr.get_value().accept(self)
    }

    // TODO: Forward declarations
    fn check_prototype(&mut self, ast: &dyn Ast<Expr>) -> bool {
        let expr: &PrototypeExpr = ast.as_impl().get_kind().to_prototype().unwrap();
        for arg in expr.get_args().iter() {
            let sym = arg.get_symbol().unwrap();
            if self.scope.contains_symbol(&sym, self.options) {
                eprintln!("Redeclaration of symbol '{}' {}", sym, arg.get_loc());
                return false;
            }
        }
        let sym = ast.get_symbol().unwrap();
        if self.scope.contains_symbol(&sym, self.options) {
            eprintln!("Redeclaration of function '{}' {}", sym, ast.get_loc());
            return false;
        }
        true
    }

    fn check_return(&mut self, ast: &dyn Ast<Expr>) -> bool {
        let expr: &ReturnExpr = ast.as_impl().get_kind().to_return().unwrap();
        match expr.get_value() {
            None        => true,
            Some(value) => value.accept(self),
        }
    }

    fn check_transpose(&mut self, ast: &dyn Ast<Expr>) -> bool {
        let expr: &TransposeExpr = ast.as_impl().get_kind().to_transpose().unwrap();
        let sym = ast.get_symbol().unwrap();
        if !self.scope.contains_symbol(&sym, self.options) {
            eprintln!("Expected to find symbol '{}' in scope {}", sym, ast.get_loc());
            return false;
        }
        expr.get_value().accept(self)
    }

    fn check_var(&mut self, ast: &dyn Ast<Expr>) -> bool {
        let _expr: &VarExpr = ast.as_impl().get_kind().to_var().unwrap();
        let sym = ast.get_symbol().unwrap();
        if !self.scope.contains_symbol(&sym, self.options) {
            eprintln!("Expected to find symbol '{}' in scope {}", sym, ast.get_loc());
            false
        } else {
            true
        }
    }

    fn check_var_decl(&mut self, ast: &dyn Ast<Expr>) -> bool {
        let expr: &VarDeclExpr = ast.as_impl().get_kind().to_var_decl().unwrap();
        if !expr.get_value().accept(self) {
            return false;
        }
        let sym = ast.get_symbol().unwrap();
        if !self.scope.add_symbol(&sym, self.options) {
            eprintln!("Redeclaration of symbol '{}' {}", sym, ast.get_loc());
            false
        } else {
            true
        }
    }
}

impl <'a> AstVisitor<Expr> for DeclCheck<'a> {
    fn visit(&mut self, ast: &dyn Ast<Expr>) -> bool {
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
