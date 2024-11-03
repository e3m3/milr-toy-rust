// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

use std::collections::HashMap;

use crate::ast;
use crate::exit_code;
use crate::options;

use ast::Ast;
use ast::AcceptGen;
use ast::AstGenerator;
use ast::Binop;
use ast::CheckUnderspecified;
use ast::Dims;
use ast::Expr;
use ast::ExprKindID;
use ast::GenResult;
use ast::SharedValue;
use ast::Shape;
use ast::Symbol;
use ast::TDim;
use ast::Type;
use ast::TypeBase;
use ast::TypeMap;
use ast::TypeSignature;
use ast::TypeTensor;
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

#[derive(Clone,Default)]
pub struct ParamIter {
    param: Box<Type>,
    params: Vec<Type>,
    index: usize,
}

type State = TypeCheckState;

#[derive(Default)]
struct TypeCheckState {
    ast_cache: HashMap<Symbol, SharedValue>,
    map: TypeMap,
    module: Option<Symbol>,
}

pub struct TypeCheck<'a> {
    options: &'a RunOptions,
    state: State,
}

#[derive(Clone,Copy,Default,PartialEq,PartialOrd)]
pub struct TypeSignatureMatchScore(f64);

impl <'a> TypeCheck<'a> {
    pub fn new(options: &'a RunOptions) -> Self {
        TypeCheck{options, state: Default::default()}
    }

    pub fn get_type_map_mut(&mut self) -> &mut TypeMap {
        self.state.get_type_map_mut()
    }

    fn process_binop(
        &mut self,
        ast: &dyn Ast<Expr>,
        _acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &BinopExpr = ast.as_impl().get_kind().to_binop().unwrap();
        let sym = expr.get_symbol();
        let op = expr.get_op();
        self.emit_message(format!("Processing Binop '{}' {}", op, ast.get_location()));
        let t_lhs = expr.get_lhs().accept_gen(self, None)?;
        let t_rhs = expr.get_rhs().accept_gen(self, None)?;
        Self::process_binop_arith(&t_lhs)?;
        Self::process_binop_arith(&t_rhs)?;
        match op {
            Binop::Add      => self.process_binop_match(expr, &sym, op, &t_lhs, &t_rhs),
            Binop::Sub      => self.process_binop_match(expr, &sym, op, &t_lhs, &t_rhs),
            Binop::Mul      => self.process_binop_match(expr, &sym, op, &t_lhs, &t_rhs),
            Binop::MatMul   => self.process_binop_match_mat_mul(expr, &sym, op, &t_lhs, &t_rhs),
            Binop::Div      => Err(format!("Binary op '{}' is unimplemented", op)), // TODO
        }
    }

    fn process_binop_arith(t: &Type) -> Result<(), String> {
        match t {
            Type::Scalar(_) => Ok(()),
            Type::Tensor(_) => Ok(()),
            Type::Undef     => Ok(()),
            Type::Unit      => Err("Cannot compute binary operation on type '()'".to_string()),
            Type::Sig(_)    => Err("Cannot compute binary operation on type 'fn(_)'".to_string()),
        }
    }

    fn process_binop_match(
        &mut self,
        expr: &BinopExpr,
        sym: &Symbol,
        op: Binop,
        t_lhs: &Type,
        t_rhs: &Type,
    ) -> GenResult<Type> {
        assert!(op != Binop::MatMul);
        if t_lhs == t_rhs {
            let t_proto = Type::new_signature(TypeSignature::new_prototype(
                vec![t_lhs.clone(), t_rhs.clone()])
            );
            let t = t_lhs.clone();
            self.specialize_function(sym, &t_proto, &t, false)?;
            self.emit_message(format!("Is type '{}'", t));
            Ok(t)
        } else if t_lhs.is_underspecified() && !t_rhs.is_underspecified() {
            let t_proto = Type::new_signature(TypeSignature::new_prototype(
                vec![t_rhs.clone(), t_rhs.clone()])
            );
            let t = t_rhs.clone();
            let sym_lhs = expr.get_lhs().get_symbol().unwrap();
            self.emit_message(format!("Inferring type for '{}' to '{}'", sym_lhs, t));
            self.state.get_type_map_mut().add_type(&sym_lhs, &t)?;
            self.specialize_function(sym, &t_proto, &t, false)?;
            self.emit_message(format!("Is type '{}'", t));
            Ok(t)
        } else if !t_lhs.is_underspecified() && t_rhs.is_underspecified() {
            let t_proto = Type::new_signature(TypeSignature::new_prototype(
                vec![t_lhs.clone(), t_lhs.clone()])
            );
            let t = t_lhs.clone();
            let sym_rhs = expr.get_rhs().get_symbol().unwrap();
            self.emit_message(format!("Inferring type for '{}' to '{}'", sym_rhs, t));
            self.state.get_type_map_mut().add_type(&sym_rhs, &t)?;
            self.specialize_function(sym, &t_proto, &t, false)?;
            self.emit_message(format!("Is type '{}'", t));
            Ok(t)
        } else {
            Err(format!(
                "Binop {} has mismatched lhs '{}' and rhs '{}' types", op, t_lhs, t_rhs
            ))
        }
    }

    fn process_binop_match_mat_mul(
        &mut self,
        expr: &BinopExpr,
        sym: &Symbol,
        op: Binop,
        t_lhs: &Type,
        t_rhs: &Type,
    ) -> GenResult<Type> {
        assert!(op == Binop::MatMul);
        let (t_lhs, t_rhs, t) = if t_lhs.is_undef() && t_rhs.is_undef() {
            (t_lhs.clone(), t_rhs.clone(), Type::default())
        } else if t_lhs.is_underspecified() && t_rhs.is_underspecified() {
            let t = match t_lhs.get_type() {
                Some(t) => {
                    let tt = Type::new_tensor(TypeTensor::new_unranked(t));
                    let sym_rhs = expr.get_rhs().get_symbol().unwrap();
                    self.state.get_type_map_mut().add_type(&sym_rhs, &tt)?;
                    self.emit_message(format!("Inferring type for '{}' to '{}'", sym_rhs, tt));
                    tt
                },
                None    => match t_rhs.get_type() {
                    Some(t) => {
                        let tt = Type::new_tensor(TypeTensor::new_unranked(t));
                        let sym_lhs = expr.get_lhs().get_symbol().unwrap();
                        self.state.get_type_map_mut().add_type(&sym_lhs, &tt)?;
                        self.emit_message(format!("Inferring type for '{}' to '{}'", sym_lhs, tt));
                        tt
                    },
                    None    => Type::new_tensor(TypeTensor::new_unranked(TypeBase::F64)), // Default to F64
                },
            };
            (t_lhs.clone(), t_rhs.clone(), t)
        } else if t_lhs.is_underspecified() {
            let t_lhs = t_rhs.transpose().unwrap();
            let sym_lhs = expr.get_lhs().get_symbol().unwrap();
            self.state.get_type_map_mut().add_type(&sym_lhs, &t_lhs)?;
            self.emit_message(format!("Inferring type for '{}' to '{}'", sym_lhs, t_lhs));
            let t = Type::mat_mul(&t_lhs, t_rhs);
            (t_lhs.clone(), t_rhs.clone(), t)
        } else if t_rhs.is_underspecified() {
            let t_rhs = t_lhs.transpose().unwrap();
            let sym_rhs = expr.get_rhs().get_symbol().unwrap();
            self.state.get_type_map_mut().add_type(&sym_rhs, &t_rhs)?;
            self.emit_message(format!("Inferring type for '{}' to '{}'", sym_rhs, t_rhs));
            let t = Type::mat_mul(t_lhs, &t_rhs);
            (t_lhs.clone(), t_rhs.clone(), t)
        } else {
            let t = Type::mat_mul(t_lhs, t_rhs);
            (t_lhs.clone(), t_rhs.clone(), t)
        };
        let t_proto = Type::new_signature(TypeSignature::new_prototype(vec![t_lhs, t_rhs]));
        self.specialize_function(sym, &t_proto, &t, false)?;
        self.emit_message(format!("Is type '{}'", t));
        Ok(t)
    }

    /// The point at which a function is processed may not have enough context in the TypeMap to
    /// determine the types of a functions parameters, nor enough context to determine the
    /// type signature specializations required for callees within the function body:
    /// ```mlp
    /// def multiply_transpose(a, b) {
    ///     return transpose(a) * transpose(b)
    /// }
    /// ```
    /// Here `a` and `b` can only be inferred to be undef.
    /// The specialization of `multiply_transpose(_)` (and thus the types of `a` and `b`) will be
    /// determined at a later point where there is a call to `multiply_transpose(_)`.
    /// This is done by caching the AST upon first discovery of a function defintion, and then
    /// repeating the processing with a parameter iterator containing the types for the values
    /// used at call-site to the function.
    fn process_call(
        &mut self,
        ast: &dyn Ast<Expr>,
        _acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &CallExpr = ast.as_impl().get_kind().to_call().unwrap();
        self.emit_message(format!("Processing Call '{}' {}", expr.get_callee(), ast.get_location()));
        let sym = expr.get_symbol();
        let ast_cached = self.state.get_ast_from_cache(&sym);
        if ast_cached.is_some() {
            self.emit_message(format!("Found symbol '{}' in AST cache", sym));
        }
        let mut arg_types: Vec<Type> = Vec::new();
        if expr.get_args().is_empty() {
            arg_types.push(Type::new_unit());
        } else {
            for arg in expr.get_args().iter() {
                let t = arg.accept_gen(self, None)?;
                arg_types.push(t);
            }
        }
        if ast_cached.is_some() {
            let t = self.state.get_type_map().match_type_signature(&sym, &arg_types)?;
            assert!(t.is_signature());
            self.emit_message(format!("Getting specialization from '{}' for '{}'", sym, t));
            let t = if t.is_underspecified() {
                self.emit_message(format!("Type for '{}' is underspecified", sym));
                let mut param_iter = ParamIter::new(&arg_types);
                ast_cached.unwrap().accept_gen(self, Some(&mut param_iter))?
            } else {
                t
            };
            match t {
                Type::Sig(ts)   => {
                    let t_ret = ts.get_return();
                    self.emit_message(format!("Is type '{}'", t_ret));
                    Ok(t_ret.clone())
                },
                _               => Err("Expected signature type".to_string()),
            }
        } else {
            Err(format!("Attempted to process Call '{}' for which there is no known signature", sym))
        }
    }

    /// Though shadow symbols are not allowed, dangling symbols may occur if a symbol is defined
    /// within the body of a function, and then defined later (as or within another function):
    /// ```mlp
    /// def foo() {
    ///     var bar<> = [1, 2];
    ///     var foo<> = [2, 3]; // `foo()` is not yet in scope, but will be after processing body
    ///     return bar + foo;
    /// }
    /// ```
    /// This issue is resolved by suffixing function symbols to differentiate them from variables.
    fn process_function(
        &mut self,
        ast: &dyn Ast<Expr>,
        acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &FunctionExpr = ast.as_impl().get_kind().to_function().unwrap();
        let sym = expr.get_symbol();
        self.emit_message(format!("Processing Function '{}' {}", expr.get_name(), ast.get_location()));
        let t_params: Type = expr.get_prototype().accept_gen(self, acc)?;
        self.state.push_function(&t_params);
        let mut found_terminator = false;
        for value in expr.get_body().iter() {
            let _t_value: Type = value.accept_gen(self, None)?;
            if Self::is_terminator(value.as_impl()) {
                found_terminator = true;
            }
        }
        let t_ret = if found_terminator {
            self.state.pop_terminator().unwrap()
        } else {
            self.emit_message("Found no terminators".to_string());
            Type::new_unit()
        };
        let t_params: Type = if t_params.is_underspecified() {
            // Retry prototype generation with type inference if originally underspecified
            self.state.pop_function();
            let mut param_iter = ParamIter::from_prototype(expr.get_prototype(), self.state.get_type_map());
            let t_params = expr.get_prototype().accept_gen(self, Some(&mut param_iter))?;
            self.state.push_function(&t_params);
            t_params
        } else {
            t_params
        };
        self.specialize_function(&sym, &t_params, &t_ret, true)
    }

    fn process_literal(
        &mut self,
        ast: &dyn Ast<Expr>,
        _acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &LiteralExpr = ast.as_impl().get_kind().to_literal().unwrap();
        self.emit_message(format!("Processing Literal {}", ast.get_location()));
        let shape = expr.get_shape();
        self.emit_message(format!("Asserting shape is '{}'", shape));
        let mut dims: Dims = Dims::new();
        dims.push(expr.get_values().len() as TDim);
        for (i, value) in expr.get_values().iter().enumerate() {
            let t = value.accept_gen(self, None)?;
            if i == 0 {
                match t {
                    Type::Scalar(_ts)   => (),
                    Type::Tensor(tt)    => dims.append(&mut tt.get_shape().get().clone()),
                    _                   => {
                        eprintln!("Unexpected type '{}' in literal expression", t);
                        exit(ExitCode::SemanticError);
                    },
                }
            }
        }
        let shape_found = Shape::new(&dims);
        if *shape == shape_found {
            self.emit_message(format!("Shape '{}' for literal is correct", shape));
            let t = Type::new_tensor(TypeTensor::new(shape, TypeBase::F64));
            self.emit_message(format!("Is type '{}'", t));
            Ok(t)
        } else {
            Err(format!("Shape '{}' does not match expression shape '{}'", shape, shape_found))
        }
    }

    /// Since scopes are not explicitly handled by the TypeMap structure, dangling symbols
    /// from previously processed AST components may be present in the map.
    /// However, since DeclCheck is completed before TypeCheck, we know that any possibly
    /// conflicting symbols defined at a later time are legal.
    /// Instead of modifying the TypeMap structure when scopes close, simply continue adding
    /// new types for symbols and use the latest definition via `TypeMap::get_latest_type(_)`.
    /// ```mlp
    /// def test() {
    ///     var bar<> = [1, 2];
    ///     var foo<> = [2, 3];                         // Symbol 'foo' is inserted with '<2xf64>'
    ///     return bar + foo;
    /// }
    /// def main() {
    ///     var foo<3,2> = [[1, 2], [3, 4], [5, 6]];    // Symbol 'foo' is inserted with '<3x2xf64>'
    ///
    ///     print(test());                              // Print is specialized for type signature
    ///                                                 // 'fn(<2xf64> -> ())'
    ///     print(foo);                                 // Print is specialized for type signature
    /// }                                               // 'fn(<3x2xf64> -> ())'
    /// ```
    fn process_module(
        &mut self,
        ast: &dyn Ast<Expr>,
        _acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &ModuleExpr = ast.as_impl().get_kind().to_module().unwrap();
        self.emit_message(format!("Running TypeCheck on module '{}'", expr.get_name()));
        self.state.module = Some(expr.get_symbol());
        for function in expr.get_functions().iter() {
            let _t = function.accept_gen(self, None)?;
            let sym = function.get_symbol().unwrap();
            self.emit_message(format!("Adding symbol '{}' to AST cache", sym));
            self.state.add_ast_in_cache(&sym, function);
        }
        self.state.module = None;
        Ok(Type::new_unit())
    }

    fn process_number(
        &mut self,
        ast: &dyn Ast<Expr>,
        _acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &NumberExpr = ast.as_impl().get_kind().to_number().unwrap();
        self.emit_message(format!("Processing Number '{}' {}", expr.get_value(), ast.get_location()));
        let t: Type = Type::new_scalar(TypeBase::new_f64());
        self.emit_message(format!("Is type '{}'", t));
        Ok(t)
    }

    fn process_param(
        &mut self,
        ast: &dyn Ast<Expr>,
        acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &VarExpr = ast.as_impl().get_kind().to_var().unwrap();
        assert!(expr.is_param());
        let sym = expr.get_symbol();
        if acc.is_none() {
            // Prototype is underspecified; default arguments to undef
            self.emit_message(format!("Found parameter '{}' {}", sym, ast.get_location()));
            let t = Type::default();
            self.state.get_type_map_mut().add_type(&sym, &t)?;
            Ok(t)
        } else {
            let t = acc.unwrap().next().unwrap();
            self.emit_message(format!("Found parameter '{}' with type '{}' {}", sym, t, ast.get_location()));
            self.state.get_type_map_mut().add_type(&sym, &t)?;
            Ok(*t)
        }
    }

    fn process_print(
        &mut self,
        ast: &dyn Ast<Expr>,
        _acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &PrintExpr = ast.as_impl().get_kind().to_print().unwrap();
        self.emit_message(format!("Processing Print {}", ast.get_location()));
        let sym = expr.get_symbol();
        let t = expr.get_value().accept_gen(self, None)?;
        let t_ret = Type::new_unit();
        let t_proto = Type::new_signature(TypeSignature::new_prototype(vec![t]));
        self.specialize_function(&sym, &t_proto, &t_ret, false)?;
        Ok(t_ret)
    }

    fn process_prototype(
        &mut self,
        ast: &dyn Ast<Expr>,
        acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &PrototypeExpr = ast.as_impl().get_kind().to_prototype().unwrap();
        let sym = expr.get_symbol();
        let mut arg_types: Vec<Type> = Vec::new();
        if expr.is_empty() {
            arg_types.push(Type::new_unit());
        } else if acc.is_none() {
            let args = expr.get_args();
            for arg in args.iter() {
                let t_param = arg.accept_gen(self, None)?;
                arg_types.push(t_param);
            }
        } else {
            let mut param_iter = acc.unwrap();
            let args = expr.get_args();
            if args.len() != param_iter.len() {
                Err(format!("Expected '{}' arguments for prototype '{}'", args.len(), sym))?
            }
            for arg in args.iter() {
                let t_param = arg.accept_gen(self, Some(&mut param_iter))?;
                arg_types.push(t_param);
            }
        }
        let t = Type::new_signature(TypeSignature::new_prototype(arg_types));
        Ok(t)
    }

    fn process_return(
        &mut self,
        ast: &dyn Ast<Expr>,
        _acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &ReturnExpr = ast.as_impl().get_kind().to_return().unwrap();
        self.emit_message(format!("Processing Return {}", ast.get_location()));
        let t = if expr.is_empty() {
            Type::new_unit()
        } else {
            expr.get_value().as_ref().unwrap().accept_gen(self, None)?
        };
        self.emit_message(format!("Return type is '{}'", t));
        self.state.push_terminator(&t);
        Ok(t)
    }

    fn process_transpose(
        &mut self,
        ast: &dyn Ast<Expr>,
        _acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &TransposeExpr = ast.as_impl().get_kind().to_transpose().unwrap();
        self.emit_message(format!("Processing Transpose {}", ast.get_location()));
        let sym = expr.get_symbol();
        let t = expr.get_value().accept_gen(self, None)?;
        let t_ret = match t.transpose() {
            Some(tt)    => tt,
            None        => Type::default(),
        };
        let t_proto = Type::new_signature(TypeSignature::new_prototype(vec![t]));
        self.specialize_function(&sym, &t_proto, &t_ret, false)?;
        self.emit_message(format!("Is type '{}'", t_ret));
        Ok(t_ret)
    }

    fn process_var(
        &mut self,
        ast: &dyn Ast<Expr>,
        acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &VarExpr = ast.as_impl().get_kind().to_var().unwrap();
        if expr.is_param() {
            return self.process_param(ast, acc);
        }
        let sym = expr.get_symbol();
        self.emit_message(format!("Processing Var '{}' {}", sym, ast.get_location()));
        let type_map = self.state.get_type_map();
        if type_map.contains_symbol(&sym) {
            let t = type_map.get_latest_type(&sym)?;
            self.emit_message(format!("Is type '{}'", t));
            Ok(t)
        } else {
            self.emit_message(format!("Found no type for '{}'", sym));
            Ok(Type::default())
        }
    }

    fn process_var_decl(
        &mut self,
        ast: &dyn Ast<Expr>,
        _acc: Option<&mut ParamIter>,
    ) -> GenResult<Type> {
        let expr: &VarDeclExpr = ast.as_impl().get_kind().to_var_decl().unwrap();
        let sym = expr.get_symbol();
        self.emit_message(format!("Processing VarDecl '{}' {}", sym, ast.get_location()));
        self.emit_message(format!("Inferring type for '{}'", sym));
        let t = if expr.is_shapeless() || expr.get_shape().is_empty() {
            expr.get_value().accept_gen(self, None)?
        } else {
            let shape = expr.get_shape();
            self.emit_message(format!("Asserting shape is '{}' for '{}'", shape, sym));
            let t = expr.get_value().accept_gen(self, None)?;
            match t {
                Type::Tensor(ref tt)    => if shape.matches(tt.get_shape()) {
                    self.emit_message(format!("Shape '{}' for '{}' is correct", shape, sym));
                    Type::new_tensor(TypeTensor::new(shape, TypeBase::F64))
                } else {
                    Err(format!("Shape '{}' does not match expression for '{}'", shape, sym))?
                },
                _                       => {
                    Err(format!("Shape '{}' does not match expression for '{}'", shape, sym))?
                }
            }
        };
        self.emit_message(format!("Setting type for '{}' to '{}'", sym, t));
        self.state.get_type_map_mut().add_type(&sym, &t)?;
        Ok(t)
    }

    fn is_terminator(ast: &dyn Ast<Expr>) -> bool {
        ast.as_impl().is_one_of(&[ExprKindID::Return])
    }

    fn emit_message(&self, message: String) -> () {
        if self.options.is_verbose(VerboseMode::Sem) {
            eprintln!("{}", message);
        }
    }

    fn specialize_function(
        &mut self,
        sym: &Symbol,
        t_prototype: &Type,
        t_ret: &Type,
        assert_prototype: bool,
    ) -> GenResult<Type> {
        if assert_prototype {
            assert!(self.state.pop_function().unwrap() == *t_prototype);
        }
        let params = match t_prototype {
            Type::Sig(t)    => t.get_params(),
            _               => Err("Expected prototype for specialization".to_string())?,
        };
        let t = Type::new_signature(TypeSignature::new(params.clone(), t_ret.clone()));
        self.emit_message(format!("Adding specialization to '{}' for '{}'", sym, t));
        self.state.get_type_map_mut().add_type(&sym, &t)?;
        Ok(t)
    }
}

impl TypeCheckState {
    pub fn add_ast_in_cache(&mut self, sym: &Symbol, ast: &SharedValue) -> () {
        if !self.ast_cache.contains_key(sym) {
            if self.ast_cache.insert(sym.clone(), ast.clone()).is_some() {
                eprintln!("Attempted to overwrite symbol '{}' in AST cache", sym);
                exit(ExitCode::SemanticError);
            }
        }
    }

    pub fn get_ast_from_cache(&self, sym: &Symbol) -> Option<SharedValue> {
        self.ast_cache.get(sym).cloned()
    }

    pub fn get_type_map(&self) -> &TypeMap {
        &self.map
    }

    pub fn get_type_map_mut(&mut self) -> &mut TypeMap {
        &mut self.map
    }

    pub fn push_function(&mut self, t: &Type) -> () {
        match self.map.add_type(&Self::symbol_function(), t) {
            Ok(())      => (),
            Err(msg)    => {
                eprintln!("Failed to push function type '{}' to stack: {}", t, msg);
                exit(ExitCode::SemanticError);
            }
        }
    }

    pub fn push_terminator(&mut self, t: &Type) -> () {
        match self.map.add_type(&Self::symbol_terminator(), t) {
            Ok(())      => (),
            Err(msg)    => {
                eprintln!("Failed to push terminator type '{}' to stack: {}", t, msg);
                exit(ExitCode::SemanticError);
            }
        }
    }

    pub fn pop_function(&mut self) -> Option<Type> {
        match self.map.pop_latest_type(&Self::symbol_function()) {
            Ok(t)       => t,
            Err(msg)    => {
                eprintln!("Failed to pop function type from stack: {}", msg);
                exit(ExitCode::SemanticError);
            },
        }
    }

    pub fn pop_terminator(&mut self) -> Option<Type> {
        match self.map.pop_latest_type(&Self::symbol_terminator()) {
            Ok(t)       => t,
            Err(msg)    => {
                eprintln!("Failed to pop terminator type from stack: {}", msg);
                exit(ExitCode::SemanticError);
            },
        }
    }

    fn symbol_function() -> Symbol {
        Symbol::new_internal("function_stack")
    }

    fn symbol_terminator() -> Symbol {
        Symbol::new_internal("terminator_stack")
    }
}

impl ParamIter {
    pub fn new(params: &Vec<Type>) -> Self {
        ParamIter{param: Box::new(Type::default()), params: params.clone(), index: Default::default()}
    }

    pub fn from_prototype(proto: &SharedValue, type_map: &TypeMap) -> Self {
        let expr = proto.get_kind().to_prototype().unwrap();
        let params: Vec<Type> = expr.get_args()
            .iter()
            .map(|a| type_map.get_latest_type(&a.get_symbol().unwrap()).unwrap_or(Type::default()))
            .collect();
        Self::new(&params)
    }

    pub fn get(&self, index: usize) -> &Type {
        match self.params.get(index) {
            Some(t) => t,
            None    => {
                eprintln!("Expected index '{}' in bounds for parameter iterator", index);
                exit(ExitCode::SemanticError);
            },
        }
    }

    pub fn get_params(&self) -> &Vec<Type> {
        &self.params
    }

    pub fn len(&self) -> usize {
        self.params.len()
    }
}

impl Iterator for ParamIter {
    type Item = Box<Type>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.len() {
            self.param = Box::new(self.get(self.index).clone());
            self.index += 1;
            Some(self.param.clone())
        } else {
            None
        }
    }
}

impl TypeMap {
    /// Get the closest matching signature from the type list for the given symbol.
    pub fn match_type_signature(&self, sym: &Symbol, params: &Vec<Type>) -> Result<Type, String> {
        let v = self.get_types(sym)?;
        let mut t_match = Type::default();
        let mut match_score = TypeSignatureMatchScore::default();
        for t in v.iter() {
            match t {
                Type::Sig(ts)   => {
                    let match_score_ts = ts.match_params(params);
                    // Take the earliest best match
                    // This should take underspecified signatures over wrong signatures
                    if match_score_ts > match_score {
                        t_match = t.clone();
                        match_score = match_score_ts;
                    }
                },
                _               => (),
            };
        };
        if t_match.is_undef() {
            Err(format!(
                "Found no suitable type match for symbol '{}' for parameters", sym
            ))
        } else {
            Ok(t_match)
        }
    }
}

impl TypeSignature {
    /// Reward matches with the right number of parameters with BIAS
    const BIAS: f64 = 0.00000001;

    /// Walk through the given parameters, and compare them with this signature (self).
    /// If the lengths don't match, the score for the given parameters is the same
    /// as an undef signature, which will not be considered.
    /// Give signatures with the right number of parameters for a given symbol a small
    /// bias to remove the deadlock with an an undef signature (the default for the outer
    /// loop; caller of `match_params(_)`).
    pub fn match_params(&self, params: &Vec<Type>) -> TypeSignatureMatchScore {
        let num_params = self.get_params().len();
        if num_params != params.len() {
            return TypeSignatureMatchScore::default();
        }
        let mut matching_params: usize = 0;
        for (i, param) in self.get_params().iter().enumerate() {
            let param_other = params.get(i).unwrap();
            eprintln!("Checking params '{}' and '{}'", param, param_other);
            if param == param_other {
                matching_params += 1;
            }
        }
        let score = f64::from(matching_params as u16)/f64::from(num_params as u16) + Self::BIAS;
        TypeSignatureMatchScore::new(score)
    }
}

impl TypeSignatureMatchScore {
    pub fn new(score: f64) -> Self {
        TypeSignatureMatchScore{0: score}
    }
}

impl <'a> AstGenerator<Expr, Option<&mut ParamIter>, Type> for TypeCheck<'a> {
    fn gen(&mut self, ast: &dyn Ast<Expr>, acc: Option<&mut ParamIter>) -> GenResult<Type> {
        let expr = ast.as_impl();
        match expr.get_kind_id() {
            ExprKindID::Binop       => self.process_binop(ast, acc),
            ExprKindID::Call        => self.process_call(ast, acc),
            ExprKindID::Function    => self.process_function(ast, acc),
            ExprKindID::Literal     => self.process_literal(ast, acc),
            ExprKindID::Module      => self.process_module(ast, acc),
            ExprKindID::Number      => self.process_number(ast, acc),
            ExprKindID::Print       => self.process_print(ast, acc),
            ExprKindID::Prototype   => self.process_prototype(ast, acc),
            ExprKindID::Return      => self.process_return(ast, acc),
            ExprKindID::Transpose   => self.process_transpose(ast, acc),
            ExprKindID::Var         => self.process_var(ast, acc),
            ExprKindID::VarDecl     => self.process_var_decl(ast, acc),
            ExprKindID::Unset       => {
                eprintln!("Unexpected AST of kind Unset {}", ast.get_location());
                exit(ExitCode::SemanticError);
            },
        }
    }
}
