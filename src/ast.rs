// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

//extern crate llvm_sys as llvm;
//use llvm::prelude::LLVMValueRef;

use crate::exit_code;
use exit_code::exit;
use exit_code::ExitCode;

use std::fmt;
use std::mem;
use std::path::Path;
use std::slice;
use std::vec::Vec;

////////////////////////////////////////
//  Ast trait section
////////////////////////////////////////

//pub type GenResult = Result<LLVMValueRef, &'static str>;

//pub trait AstGenerator {
//    fn visit(&mut self, ast: &dyn Ast) -> GenResult;
//}

pub trait AstVisitor {
    fn visit(&mut self, ast: &dyn Ast) -> bool;
}

pub trait Ast: fmt::Display {
    fn accept(&self, visitor: &mut dyn AstVisitor) -> bool;
    fn get_kind_id(&self) -> ExprKindID;
    fn get_kind(&self) -> &ExprKind;
    fn get_loc(&self) -> &Location;
    fn isa(&self, id: ExprKindID) -> bool;
    //fn accept_gen(&self, visitor: &mut dyn AstGenerator) -> GenResult;
}

pub trait ExprDisplay {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result;

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.expr_fmt(f, 0, &Default::default())
    }

    fn indent(depth: usize) -> String {
        let mut s = String::new();
        for _ in 0..depth {
            s += "  ";
        }
        s
    }
}

pub struct Expr {
    kind: ExprKind,
    id: ExprKindID,
    loc: Location,
}

#[repr(u8)]
#[derive(Clone,Copy,PartialEq)]
pub enum ExprKindID {
    Binop,
    Call,
    Function,
    Literal,
    Module,
    Number,
    Print,
    Prototype,
    Return,
    Transpose,
    Unset,
    Var,
    VarDecl,
}

pub enum ExprKind {
    Binop(BinopExpr),
    Call(CallExpr),
    Function(FunctionExpr),
    Literal(LiteralExpr),
    Module(ModuleExpr),
    Number(NumberExpr),
    Print(PrintExpr),
    Prototype(PrototypeExpr),
    Return(ReturnExpr),
    Transpose(TransposeExpr),
    Unset(),
    Var(VarExpr),
    VarDecl(VarDeclExpr),
}

////////////////////////////////////////
//  Struct section
////////////////////////////////////////

pub type TDim = i64;
pub type TMlp = f64;

#[repr(u8)]
#[derive(Clone,Copy,PartialEq)]
pub enum Binop {
    Add,
    Div,
    Mul,
    Sub,
}

pub struct BinopExpr {
    op: Binop,
    lhs: Value,
    rhs: Value,
}

#[derive(Clone,Copy,PartialEq,PartialOrd)]
pub struct BinopPrecedence(u8);

pub struct CallExpr {
    args: Values,
    name: String,
}

pub type Dims = Vec<TDim>;

pub struct FunctionExpr {
    proto: Value,
    values: Values,
}

pub struct LiteralExpr {
    shape: Shape,
    values: Values,
}

#[derive(Clone)]
pub struct Location {
    file_name: String,
    line_no: usize,
    col_no: usize,
}

pub struct ModuleExpr {
    name: String,
    values: Values,
}

#[derive(Clone)]
pub struct NumberExpr {
    value: TMlp,
}

pub struct PrintExpr {
    value: Value,
}

pub struct PrototypeExpr {
    name: String,
    values: Values,
}

pub struct ReturnExpr {
    value: Option<Value>
}

pub struct TransposeExpr {
    value: Value,
}

#[derive(Clone,PartialEq)]
pub struct Shape(Dims);

pub type Value = Box<Expr>;

pub struct Values(Vec<Value>);

#[derive(Clone)]
pub struct VarExpr {
    name: String,
}

pub struct VarDeclExpr {
    name: String,
    shape: Shape,
    value: Value,
}

////////////////////////////////////////
//  Struct implementation section
////////////////////////////////////////

pub fn binop_from_str(op: &str) -> Binop {
    match op {
        "-" => Binop::Sub,
        "+" => Binop::Add,
        "/" => Binop::Div,
        "*" => Binop::Mul,
        _   => {
            eprintln!("Unexpected op string '{}'", op);
            exit(ExitCode::AstError);
        },
    }
}

impl BinopExpr {
    pub fn new(op: Binop, lhs: Value, rhs: Value) -> Self {
        BinopExpr{op, lhs, rhs}
    }

    pub fn get_lhs(&self) -> &Value {
        &self.lhs
    }

    pub fn get_op(&self) -> Binop {
        self.op
    }

    pub fn get_precedence(&self) -> BinopPrecedence {
        BinopPrecedence::new(self.op)
    }

    pub fn get_rhs(&self) -> &Value {
        &self.rhs
    }
}

impl BinopPrecedence {
    pub fn new(op: Binop) -> Self {
        BinopPrecedence{0: match op {
            Binop::Add => 20,
            Binop::Div => 40,
            Binop::Mul => 40,
            Binop::Sub => 20,
        }}
    }

    pub fn next(&self) -> Self {
        BinopPrecedence{0: self.0 + 1}
    }
}

/// Implements the lowest possible binary operator precedence
impl Default for BinopPrecedence {
    fn default() -> Self {
        BinopPrecedence{0: 0}
    }
}

impl CallExpr {
    pub fn new(name: String, args: Values) -> Self {
        CallExpr{name, args}
    }

    pub fn get_args(&self) -> &Values {
        &self.args
    }

    pub fn get_callee(&self) -> &String {
        &self.name
    }
}

impl FunctionExpr {
    pub fn new(proto: Value, values: Values) -> Self {
        FunctionExpr{proto, values}
    }

    pub fn get_body(&self) -> &Values {
        &self.values
    }

    pub fn get_prototype(&self) -> &Value {
        &self.proto
    }
}

impl LiteralExpr {
    pub fn new(shape: Shape, values: Values) -> Self {
        LiteralExpr{shape, values}
    }

    pub fn get_shape(&self) -> &Shape {
        &self.shape
    }

    pub fn get_values(&self) -> &Values {
        &self.values
    }
}

impl Location {
    pub fn new(file_name: String, line_no: usize, col_no: usize) -> Self {
        Location{file_name, line_no, col_no}
    }

    pub fn exists(&self) -> bool {
        let p = Path::new(&self.file_name);
        p.exists()
    }

    pub fn get_col(&self) -> usize {
        self.col_no
    }

    pub fn get_line(&self) -> usize {
        self.line_no
    }
}

impl Default for Location {
    fn default() -> Self {
        Location::new(String::new(), 0, 0)
    }
}

impl ModuleExpr {
    pub fn new(name: String, values: Values) -> Self {
        ModuleExpr{name, values}
    }

    pub fn get_functions(&self) -> &Values {
        &self.values
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }
}

impl Default for ModuleExpr {
    fn default() -> Self {
        ModuleExpr::new(String::new(), Values(Vec::new()))
    }
}

impl NumberExpr {
    pub fn new(value: TMlp) -> Self {
        NumberExpr{value}
    }

    pub fn get_value(&self) -> TMlp {
        self.value
    }
}

impl PrintExpr {
    pub fn new(value: Value) -> Self {
        PrintExpr{value}
    }

    pub fn get_value(&self) -> &Value {
        &self.value
    }
}

impl PrototypeExpr {
    pub fn new(name: String, values: Values) -> Self {
        PrototypeExpr{name, values}
    }

    pub fn get_args(&self) -> &Values {
        &self.values
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }
}

impl ReturnExpr {
    pub fn new(value: Option<Value>) -> Self {
        ReturnExpr{value}
    }

    pub fn get_value(&self) -> &Option<Value> {
        &self.value
    }

    pub fn is_empty(&self) -> bool {
        self.value.is_none()
    }
}

impl Shape {
    pub fn new(dims: Dims) -> Self {
        Shape{0: dims}
    }

    pub fn get(&self) -> &Dims {
        &self.0
    }
}

impl Default for Shape {
    fn default() -> Self {
        Shape::new(Vec::new())
    }
}

impl TransposeExpr {
    pub fn new(value: Value) -> Self {
        TransposeExpr{value}
    }

    pub fn get_value(&self) -> &Value {
        &self.value
    }
}

impl Values {
    pub fn new(values: Vec<Value>) -> Self {
        Values{0: values}
    }

    pub fn get(&self, i: usize) -> &Value {
        match self.0.get(i) {
            Some(value) => value,
            None        => {
                eprintln!("Expected value at pos '{}'", i);
                exit(ExitCode::AstError);
            },
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn iter(&self) -> slice::Iter<Value> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn push(&mut self, value: Value) -> () {
        self.0.push(value)
    }
}

impl Default for Values {
    fn default() -> Self {
        Values::new(Vec::new())
    }
}

impl VarExpr {
    pub fn new(name: String) -> Self {
        VarExpr{name}
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }
}

impl VarDeclExpr {
    pub fn new(name: String, shape: Shape, value: Value) -> Self {
        VarDeclExpr{name, shape, value}
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn get_shape(&self) -> &Shape {
        &self.shape
    }

    pub fn get_value(&self) -> &Value {
        &self.value
    }

    pub fn is_shapeless(&self) -> bool {
        self.shape.0.is_empty()
    }
}

////////////////////////////////////////
//  Ast implementation section
////////////////////////////////////////

impl Ast for Expr {
    fn accept(&self, visitor: &mut dyn AstVisitor) -> bool {
        visitor.visit(self)
    }

    fn get_kind(&self) -> &ExprKind {
        &self.kind
    }

    fn get_kind_id(&self) -> ExprKindID {
        self.id
    }

    fn get_loc(&self) -> &Location {
        &self.loc
    }

    fn isa(&self, id: ExprKindID) -> bool {
        self.get_kind_id() == id
    }
}

impl Expr {
    pub fn as_value(&mut self) -> Value {
        Value::new(mem::take(self))
    }

    pub fn new(kind: ExprKind, id: ExprKindID, loc: Location) -> Self {
        Expr{kind, id, loc}
    }

    /// Convenience initializer for ExprKind::Binop
    pub fn new_binop(op: Binop, lhs: Value, rhs: Value, loc: Location) -> Self {
        Expr::new(ExprKind::Binop(BinopExpr::new(op, lhs, rhs)), ExprKindID::Binop, loc)
    }

    /// Convenience initializer for ExprKind::Call
    pub fn new_call(name: String, args: Values, loc: Location) -> Self {
        Expr::new(ExprKind::Call(CallExpr::new(name, args)), ExprKindID::Call, loc)
    }

    /// Convenience initializer for ExprKind::Function
    pub fn new_function(proto: Value, values: Values, loc: Location) -> Self {
        Expr::new(ExprKind::Function(FunctionExpr::new(proto, values)), ExprKindID::Function, loc)
    }

    /// Convenience initializer for ExprKind::Literal
    pub fn new_literal(shape: Shape, values: Values, loc: Location) -> Self {
        Expr::new(ExprKind::Literal(LiteralExpr::new(shape, values)), ExprKindID::Literal, loc)
    }

    /// Convenience initializer for ExprKind::Module
    pub fn new_module(name: String, values: Values, loc: Location) -> Self {
        Expr::new(ExprKind::Module(ModuleExpr::new(name, values)), ExprKindID::Module, loc)
    }

    /// Convenience initializer for ExprKind::Number
    pub fn new_number(value: TMlp, loc: Location) -> Self {
        Expr::new(ExprKind::Number(NumberExpr::new(value)), ExprKindID::Number, loc)
    }

    /// Convenience initializer for ExprKind::Print
    pub fn new_print(value: Value, loc: Location) -> Self {
        Expr::new(ExprKind::Print(PrintExpr::new(value)), ExprKindID::Print, loc)
    }

    /// Convenience initializer for ExprKind::Prototype
    pub fn new_prototype(name: String, values: Values, loc: Location) -> Self {
        Expr::new(ExprKind::Prototype(PrototypeExpr::new(name, values)), ExprKindID::Prototype, loc)
    }

    /// Convenience initializer for ExprKind::Return
    pub fn new_return(value: Option<Value>, loc: Location) -> Self {
        Expr::new(ExprKind::Return(ReturnExpr::new(value)), ExprKindID::Return, loc)
    }

    /// Convenience initializer for ExprKind::Transpose
    pub fn new_transpose(value: Value, loc: Location) -> Self {
        Expr::new(ExprKind::Transpose(TransposeExpr::new(value)), ExprKindID::Transpose, loc)
    }

    /// Convenience initializer for ExprKind::Var
    pub fn new_var(name: String, loc: Location) ->  Self {
        Expr::new(ExprKind::Var(VarExpr::new(name)), ExprKindID::Var, loc)
    }

    /// Convenience initializer for ExprKind::VarDecl
    pub fn new_var_decl(name: String, shape: Shape, value: Value, loc: Location) -> Self {
        Expr::new(ExprKind::VarDecl(VarDeclExpr::new(name, shape, value)), ExprKindID::VarDecl, loc)
    }
}

impl Default for Expr {
    fn default() -> Self {
        Expr::new(ExprKind::Unset(), ExprKindID::Unset, Default::default())
    }
}

////////////////////////////////////////
//  Display implementation section
////////////////////////////////////////

impl fmt::Display for Binop {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Binop::Add  => "+",
            Binop::Div  => "/",
            Binop::Mul  => "*",
            Binop::Sub  => "-",
        })
    }
}

impl ExprDisplay for BinopExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        write!(f, "{}Binop: {} {}\n", indent, self.get_op(), loc)?;
        self.lhs.get_kind().expr_fmt(f, depth + 1, self.lhs.get_loc())?;
        write!(f, "\n")?;
        self.rhs.get_kind().expr_fmt(f, depth + 1, self.rhs.get_loc())
    }
}

impl ExprDisplay for CallExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        write!(f, "{}Call '{}': {}\n", indent, self.name, loc)?;
        self.args.expr_fmt(f, depth + 1, loc)
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.kind.expr_fmt(f, 0, self.get_loc())
    }
}

impl ExprDisplay for ExprKind {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        match self {
            ExprKind::Unset()           => write!(f, "Unset"),
            ExprKind::Binop(expr)       => expr.expr_fmt(f, depth, loc),
            ExprKind::Function(expr)    => expr.expr_fmt(f, depth, loc),
            ExprKind::Call(expr)        => expr.expr_fmt(f, depth, loc),
            ExprKind::Module(expr)      => expr.expr_fmt(f, depth, loc),
            ExprKind::Number(expr)      => expr.expr_fmt(f, depth, loc),
            ExprKind::Literal(expr)     => expr.expr_fmt(f, depth, loc),
            ExprKind::Print(expr)       => expr.expr_fmt(f, depth, loc),
            ExprKind::Prototype(expr)   => expr.expr_fmt(f, depth, loc),
            ExprKind::Return(expr)      => expr.expr_fmt(f, depth, loc),
            ExprKind::Transpose(expr)   => expr.expr_fmt(f, depth, loc),
            ExprKind::Var(expr)         => expr.expr_fmt(f, depth, loc),
            ExprKind::VarDecl(expr)     => expr.expr_fmt(f, depth, loc),
        }
    }
}

impl fmt::Display for ExprKindID {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            ExprKindID::Binop       => "Binop",
            ExprKindID::Call        => "Call",
            ExprKindID::Function    => "Function",
            ExprKindID::Literal     => "Literal",
            ExprKindID::Module      => "Module",
            ExprKindID::Number      => "Number",
            ExprKindID::Print       => "Print",
            ExprKindID::Prototype   => "Prototype",
            ExprKindID::Return      => "Return",
            ExprKindID::Transpose   => "Transpose",
            ExprKindID::Unset       => "Unset",
            ExprKindID::Var         => "Var",
            ExprKindID::VarDecl     => "VarDecl",
        })
    }
}

impl ExprDisplay for FunctionExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, _loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        self.proto.get_kind().expr_fmt(f, depth, self.proto.get_loc())?;
        write!(f, "\n{}Block {}\n", indent, "{")?;
        for value in self.values.0.iter() {
            value.get_kind().expr_fmt(f, depth + 1, value.get_loc())?;
            write!(f, "\n")?;
        }
        write!(f, "{}{} // Block", indent, "}")
    }
}

impl ExprDisplay for LiteralExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        write!(f, "{}Literal: {}[ {}\n", indent, self.shape, loc)?;
        self.values.expr_fmt(f, depth + 1, loc)?;
        write!(f, "\n{}] // Literal", indent)
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "@{}:{}:{}", self.file_name, self.line_no, self.col_no)
    }
}

impl ExprDisplay for ModuleExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        write!(f, "{}Module '{}':\n", indent, self.name)?;
        self.values.expr_fmt(f, depth + 1, loc)
    }
}

impl ExprDisplay for NumberExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, _loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        write!(f, "{}{}", indent, self.value)
    }
}

impl ExprDisplay for PrintExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        write!(f, "{}Print: {}\n", indent, loc)?;
        self.value.get_kind().expr_fmt(f, depth + 1, self.value.get_loc())
    }
}

impl ExprDisplay for PrototypeExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        write!(f, "{}Proto '{}' {}\n", indent, self.name, loc)?;
        write!(f, "{}Params: [", indent)?;
        self.values.expr_fmt(f, 0, loc)?;
        write!(f, "]")
    }
}

impl ExprDisplay for ReturnExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        if self.is_empty() {
            write!(f, "{}Return {}\n", indent, loc)
        } else {
            write!(f, "{}Return {}\n", indent, loc)?;
            let value = self.value.as_ref().unwrap();
            value.get_kind().expr_fmt(f, depth + 1, value.get_loc())
        }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<{}>", self.0
            .iter()
            .map(|n| { n.to_string() })
            .collect::<Vec<String>>()
            .join(",")
        )
    }
}

impl ExprDisplay for TransposeExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        write!(f, "{}Transpose: {}\n", indent, loc)?;
        self.value.get_kind().expr_fmt(f, depth + 1, self.value.get_loc())
    }
}

impl ExprDisplay for Values {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, _loc: &Location) -> fmt::Result {
        if !self.is_empty() {
            let indent = Self::indent(depth);
            let is_num = self.0.get(0).unwrap().isa(ExprKindID::Number);
            let (d, sep) = if is_num {
                write!(f, "{}", indent)?;
                (0, ", ")
            } else {
                (depth, "\n")
            };
            for (i, value) in self.0.iter().enumerate() {
                value.get_kind().expr_fmt(f, d, value.get_loc())?;
                if i < self.0.len() - 1 {
                    write!(f, "{}", sep)?;
                    if is_num && i % 16 == 15 {
                        write!(f, "\n{}", indent)?;
                    }
                }
            }
        }
        Ok(())
    }
}

impl ExprDisplay for VarExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        write!(f, "{}Var: {} {}", indent, self.name, loc)
    }
}

impl ExprDisplay for VarDeclExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        write!(f, "{}VarDecl {}{} {}\n", indent, self.name, self.shape, loc)?;
        self.value.get_kind().expr_fmt(f, depth + 1, self.value.get_loc())
    }
}
