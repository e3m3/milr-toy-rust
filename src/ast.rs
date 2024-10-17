// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

use crate::exit_code;
use exit_code::exit;
use exit_code::ExitCode;

use std::cmp;
use std::collections::HashSet;
use std::fmt;
use std::iter;
use std::mem;
use std::path::Path;
use std::slice;
use std::rc::Rc;
use std::vec::Vec;

extern crate multimap;
use multimap::MultiMap;

////////////////////////////////////////
//  Ast trait section
////////////////////////////////////////

pub trait AcceptGen<T, U, V> {
    fn accept_gen(
        &self,
        generator: &mut dyn AstGenerator<T, U, V>,
        acc: U,
    ) -> GenResult<V>;
}

pub trait AcceptVisit<T> {
    fn accept_visit(&self, visitor: &mut dyn AstVisitor<T>) -> bool;
}

pub trait Ast<T>: fmt::Display {
    fn as_impl(&self) -> &T;
    fn get_loc(&self) -> &Location;
    fn get_symbol(&self) -> Option<Symbol>;
}

pub trait AstGenerator<T, U, V> {
    fn gen(&mut self, ast: &dyn Ast<T>, acc: U) -> GenResult<V>;
}

pub trait AstVisitor<T> {
    fn visit(&mut self, ast: &dyn Ast<T>) -> bool;
}

/// Interface for checking if any component of a type has not been completely resolved
/// through type inference.
/// For tensors, the type is underspecified if the tensor is unranked.
/// Any types comprising underspecified types is also underspecified.
pub trait CheckUnderspecified {
    fn is_underspecified(&self) -> bool;
}

pub type Dims = Vec<TDim>;

pub type GenResult<T> = Result<T, String>;

#[derive(Clone,Default)]
pub struct Location {
    file_name: String,
    line_no: usize,
    col_no: usize,
}

#[derive(Clone,Default,PartialEq)]
pub struct Shape(Dims);

#[derive(Clone,Default,Eq,Hash,PartialEq,PartialOrd)]
pub struct Symbol(String);

pub type TDim   = u64;
pub type TMlp   = f64;

#[derive(Clone,Default,PartialEq)]
pub enum Type {
    #[default]
    Undef,
    Unit,
    Scalar(TypeBase),
    Sig(Box<TypeSignature>),
    Tensor(TypeTensor),
}

#[derive(Clone,Copy,PartialEq)]
pub enum TypeBase {
    F64,
}

#[derive(Clone,Default)]
pub struct TypeMap {
    functions: HashSet<Symbol>,
    map: MultiMap<Symbol, Type>,
}

#[derive(Clone,PartialEq)]
pub struct TypeSignature {
    params: Vec<Type>,
    ret: Type,
}

#[derive(Clone,PartialEq)]
pub struct TypeTensor {
    shape: Shape,
    t: TypeBase,
}

////////////////////////////////////////
//  Expr section
////////////////////////////////////////

pub trait ExprDisplay {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result;

    #[allow(dead_code)]
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

#[derive(Clone)]
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

#[derive(Clone)]
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

#[repr(u8)]
#[derive(Clone,Copy,PartialEq)]
pub enum Binop {
    Add,
    Div,
    MatMul,
    Mul,
    Sub,
}

#[derive(Clone)]
pub struct BinopExpr {
    op: Binop,
    lhs: SharedValue,
    rhs: SharedValue,
}

#[derive(Clone,Copy,PartialEq,PartialOrd)]
pub struct BinopPrecedence(u8);

#[derive(Clone)]
pub struct CallExpr {
    args: SharedValues,
    name: String,
}

#[derive(Clone)]
pub struct FunctionExpr {
    proto: SharedValue,
    values: SharedValues,
}

#[derive(Clone)]
pub struct LiteralExpr {
    shape: Shape,
    values: SharedValues,
}

#[derive(Clone,Default)]
pub struct ModuleExpr {
    name: String,
    values: SharedValues,
}

#[derive(Clone)]
pub struct NumberExpr {
    value: TMlp,
}

#[derive(Clone)]
pub struct PrintExpr {
    value: SharedValue,
}

#[derive(Clone)]
pub struct PrototypeExpr {
    name: String,
    values: SharedValues,
}

#[derive(Clone)]
pub struct ReturnExpr {
    value: Option<SharedValue>
}

#[derive(Clone)]
pub struct TransposeExpr {
    value: SharedValue,
}

pub type SharedValue = Rc<Expr>;
pub type Value = Box<Expr>;

pub struct Values(Vec<Value>);

#[derive(Clone,Default)]
pub struct SharedValues(Vec<SharedValue>);

#[derive(Clone)]
pub struct VarExpr {
    name: String,
    is_param: bool,
}

#[derive(Clone)]
pub struct VarDeclExpr {
    name: String,
    shape: Shape,
    value: SharedValue,
}

////////////////////////////////////////
//  Ast implementation section
////////////////////////////////////////

impl <U, V> AcceptGen<Expr, U, V> for Expr {
    fn accept_gen(
        &self,
        generator: &mut dyn AstGenerator<Expr, U, V>,
        acc: U,
    ) -> GenResult<V> {
        generator.gen(self, acc)
    }
}

impl AcceptVisit<Expr> for Expr {
    fn accept_visit(&self, visitor: &mut dyn AstVisitor<Expr>) -> bool {
        visitor.visit(self)
    }
}

impl Ast<Expr> for Expr {
    fn as_impl(&self) -> &Expr {
        self
    }

    fn get_loc(&self) -> &Location {
        &self.loc
    }

    fn get_symbol(&self) -> Option<Symbol> {
        self.get_kind().get_symbol()
    }
}

impl CheckUnderspecified for Type {
    fn is_underspecified(&self) -> bool {
        match self {
            Type::Undef     => true,
            Type::Unit      => false,
            Type::Scalar(_) => false,
            Type::Sig(t)    => t.is_underspecified(),
            Type::Tensor(t) => t.is_underspecified(),
        }
    }
}

impl CheckUnderspecified for TypeSignature {
    fn is_underspecified(&self) -> bool {
        for param in self.get_params().iter() {
            if param.is_underspecified() {
                return true;
            }
        }
        self.get_return().is_underspecified()
    }
}

impl CheckUnderspecified for TypeTensor {
    fn is_underspecified(&self) -> bool {
        self.is_unranked()
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

#[allow(non_snake_case)]
impl Shape {
    pub fn new(dims: &Dims) -> Self {
        Shape{0: dims.clone()}
    }

    pub fn get(&self) -> &Dims {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn rank(&self) -> usize {
        self.get().len()
    }

    /// Get a new shape with the same dimensions, but removing the first `n` dimensions.
    /// Given a shape with rank `N > 0`, the new rank of the shape is `max(1, N-n)`.
    pub fn all_but_first(&self, n: usize) -> Self {
        if self.is_empty() || n == 0 {
            return self.clone();
        }
        let N = self.rank();
        Self::new(&self.get()[cmp::max(1, N - n)..].to_vec())
    }

    /// Get a new shape with the same dimensions, but removing the final `n` dimensions.
    /// Given a shape with rank `N > 0`, the new rank of the shape is `max(1, N-n)`.
    pub fn all_but_last(&self, n: usize) -> Self {
        if self.is_empty() || n == 0 {
            return self.clone();
        }
        let N = self.rank();
        Self::new(&self.get()[..cmp::max(1, N - n)].to_vec())
    }

    /// Get a new shape with the same dimensions, but keeping the first `n` dimensions.
    /// Given a shape with rank `N > 0`, the new rank of the shape is `min(N, max(1, n))`
    pub fn first(&self, n: usize) -> Self {
        if self.is_empty() || n == 0 {
            return Self::default();
        }
        let N = self.rank();
        if n == N {
            return self.clone();
        }
        Self::new(&self.get()[..cmp::min(n, N)].to_vec())
    }

    /// Get a new shape of rank 1 with dimesnions equal to the product of the all the ranks.
    pub fn flatten(&self) -> Self {
        if self.is_empty() || self.rank() <= 1 {
            return self.clone();
        }
        Self::new(&vec![self.get().iter().product()])
    }

    /// Get a new shape with the same dimensions, but keeping the last `n` dimensions.
    /// Given a shape with rank `N > 0`, the new rank of the shape is `min(N, max(1, n))`.
    pub fn last(&self, n: usize) -> Self {
        if self.is_empty() || n == 0 {
            return Self::default();
        }
        let N = self.rank();
        if n == N {
            return self.clone();
        }
        Self::new(&self.get()[cmp::min(n, N - 1)..].to_vec())
    }

    /// Returns true if the flattened (rank 1) shape of each input are equal.
    pub fn matches(&self, other: &Self) -> bool {
        let dim: TDim = self.get().iter().product();
        let dim_other: TDim = other.get().iter().product();
        dim == dim_other
    }

    /// Get the number of matching inner dimensions of two shapes.
    /// The input shapes should be of the same rank, otherwise None.
    pub fn matching_inner_rank(&self, other: &Self) -> Option<usize> {
        if self.is_empty() || other.is_empty() || self.rank() != other.rank() {
            return None;
        } else {
            let a = self.last(1);
            let b = other.all_but_last(1);
            let n: usize = iter::zip(a.get().iter(), b.get().iter()).filter(|(a, b)| a == b).count();
            if n > 0 { Some(n) } else { None }
        }
    }

    /// Get a new shape from the outer product of this and another shape.
    pub fn outer_product(&self, other: &Self) -> Self {
        let dims: Dims = self.get().iter().chain(other.get().iter()).cloned().collect();
        Shape::new(&dims)
    }

    pub fn transpose(&self) -> Self {
        Shape::new(&self.get().iter().rev().copied().collect())
    }
}

impl Symbol {
    pub fn new(s: &str) -> Self {
        Symbol{0: s.to_string()}
    }

    pub fn new_function(s: &str) -> Self {
        Symbol::new(&format!("{}(_)", s))
    }

    pub fn new_internal(s: &str) -> Self {
        Symbol::new(&format!(",,{}", s))
    }

    pub fn is_function(&self) -> bool {
        self.0.rfind("(_)").is_some()
    }

    pub fn is_internal(&self) -> bool {
        self.0.find(",,").is_some()
    }

    #[allow(dead_code)]
    pub fn to_function(&self) -> Self {
        if self.is_function() {
            self.clone()
        } else {
            Self::new_function(&self.0)
        }
    }
}

impl Type {
    pub fn new_scalar(t: TypeBase) -> Self {
        Self::Scalar(t)
    }

    pub fn new_signature(sig: TypeSignature) -> Self {
        Self::Sig(Box::new(sig))
    }

    pub fn new_tensor(t: TypeTensor) -> Self {
        Self::Tensor(t)
    }

    pub fn new_unit() -> Self {
        Self::Unit
    }

    pub fn any_mismatch(slice: &[Type]) -> bool {
        if slice.is_empty() {
            return false;
        }
        let t = &slice[0];
        for u in slice[1..].iter() {
            if *t != *u {
                return true;
            }
        }
        false
    }

    pub fn get_type(&self) -> Option<TypeBase> {
        match self {
            Type::Undef     => None,
            Type::Unit      => None,
            Type::Scalar(t) => Some(t.clone()),
            Type::Sig(t)    => t.get_return().get_type(),
            Type::Tensor(t) => Some(t.get_type()),
        }
    }

    pub fn is_prototype(&self) -> bool {
        match self {
            Type::Sig(t)    => t.is_prototype(),
            _               => false,
        }
    }

    pub fn is_scalar(&self) -> bool {
        match self {
            Type::Scalar(_) => true,
            _               => false,
        }
    }

    pub fn is_signature(&self) -> bool {
        match self {
            Type::Sig(_)    => true,
            _               => false,
        }
    }

    pub fn is_tensor(&self) -> bool {
        match self {
            Type::Tensor(_) => true,
            _               => false,
        }
    }

    pub fn is_undef(&self) -> bool {
        *self == Type::Undef
    }

    pub fn is_unit(&self) -> bool {
        *self == Type::Unit
    }

    pub fn mat_mul(a: &Self, b: &Self) -> Self {
        eprintln!("Warning: Matrix multiplication for tensors is experimental and likely buggy");
        if a.is_tensor() && b.is_tensor() {
            let Type::Tensor(t_a) = a else { exit(ExitCode::AstError); };
            let Type::Tensor(t_b) = b else { exit(ExitCode::AstError); };
            Type::new_tensor(TypeTensor::mat_mul(t_a, t_b))
        } else {
            eprintln!("Unexpected input types '{}' and '{}' for matrix multiply", a, b);
            exit(ExitCode::AstError);
        }
    }

    pub fn transpose(&self) -> Option<Self> {
        match self {
            Type::Undef     => None,
            Type::Unit      => None,
            Type::Scalar(t) => Some(Type::new_scalar(t.clone())),
            Type::Sig(_)    => None,
            Type::Tensor(t) => Some(Self::new_tensor(t.transpose())),
        }
    }
}

impl TypeBase {
    pub fn new_f64() -> Self {
        Self::F64
    }

    pub fn is(&self, t: TypeBase) -> bool {
        *self == t
    }

    pub fn is_f64(&self) -> bool {
        *self == TypeBase::F64
    }
}

impl TypeMap {
    pub fn new() -> Self {
        TypeMap{functions: Default::default(), map: Default::default()}
    }

    pub fn add_type(&mut self, sym: &Symbol, t: &Type) -> Result<(), String> {
        if !sym.is_internal() && t.is_signature() {
            if sym.is_function() {
                self.functions.insert(sym.clone());
            } else {
                Err(format!(
                    "Attempted to specialize non-function '{}' with signature '{}'", sym, t
                ))?
            }
        }
        self.map.insert(sym.clone(), t.clone());
        Ok(())
    }

    pub fn contains_symbol(&self, sym: &Symbol) -> bool {
        self.map.contains_key(sym) && !self.map.get_vec(sym).unwrap().is_empty()
    }

    /// MultiMap preserves the insertion order for value types: Get the last in the list.
    pub fn get_latest_type(&self, sym: &Symbol) -> Result<Type, String> {
        match self.map.get_vec(sym) {
            None    => Err(format!("Expected types defined for symbol '{}'", sym)),
            Some(v) => {
                let n = v.len();
                Ok(v[n - 1].clone())
            },
        }
    }

    pub fn get_types(&self, sym: &Symbol) -> Result<&Vec<Type>, String> {
        match self.map.get_vec(sym) {
            Some(v) => Ok(v),
            None    => Err(format!("Expected types defined for symbol '{}'", sym)),
        }
    }

    pub fn iter(&self) -> multimap::Iter<'_, Symbol, Type> {
        self.map.iter()
    }

    /// MultiMap preserves the insertion order for value types: Pop and return the last in the list.
    pub fn pop_latest_type(&mut self, sym: &Symbol) -> Result<Option<Type>, String> {
        match self.map.get_vec_mut(sym) {
            None    => Err(format!("Expected types defined for symbol '{}'", sym)),
            Some(v) => Ok(v.pop()),
        }
    }

    fn is_function(&self, sym: &Symbol) -> bool {
        sym.is_function() && self.contains_symbol(sym) && self.functions.contains(sym)
    }
}

impl TypeSignature {
    pub fn new(params: Vec<Type>, ret: Type) -> Self {
        TypeSignature{params, ret}
    }

    pub fn new_prototype(params: Vec<Type>) -> Self {
        TypeSignature::new(params, Type::default())
    }

    pub fn is_prototype(&self) -> bool {
        self.get_return().is_undef()
    }

    pub fn get_params(&self) -> &Vec<Type> {
        &self.params
    }

    pub fn get_return(&self) -> &Type {
        &self.ret
    }

    pub fn has_return(&self) -> bool {
        !self.get_return().is_unit() && !self.get_return().is_undef()
    }
}

impl TypeTensor {
    pub fn new(shape: &Shape, t: TypeBase) -> Self {
        TypeTensor{shape: shape.clone(), t}
    }

    pub fn new_unranked(t: TypeBase) -> Self {
        Self::new(&Default::default(), t)
    }

    pub fn get_shape(&self) -> &Shape {
        &self.shape
    }

    pub fn get_type(&self) -> TypeBase {
        self.t 
    }

    pub fn is_unranked(&self) -> bool {
        self.get_shape().is_empty()
    }

    #[allow(non_snake_case)]
    pub fn mat_mul(a: &Self, b: &Self) -> Self {
        let t_a = a.get_type();
        let t_b = b.get_type();
        if t_a != t_b {
            eprintln!("Unexpected unmatched base types '{}' and '{}' for matrix multiply", t_a, t_b);
            exit(ExitCode::AstError);
        } else if a.is_unranked() || b.is_unranked() {
            eprintln!("Unexpected unranked tensor '{}' or '{}' for matrix multiply", a, b);
            exit(ExitCode::AstError);
        }
        let shape_a = a.get_shape();
        let shape_b = b.get_shape();
        let mut dims: Dims = Dims::new();
        let N = shape_a.rank();
        let n = match shape_a.matching_inner_rank(shape_b) {
            Some(n) => n,
            None    => {
                eprintln!("Mismatched tensor shapes '{}' and '{}' for matrix multiply", shape_a, shape_b);
                exit(ExitCode::AstError);
            },
        };
        dims.append(&mut shape_a.all_but_last(n).get().clone());
        dims.append(&mut shape_b.last(N - n).get().clone());
        Self::new(&Shape::new(&dims), t_a)
    }

    pub fn rank(&self) -> usize {
        self.get_shape().rank()
    }

    pub fn set_rank(&mut self, shape: &Shape) -> () {
        if self.is_unranked() {
            self.shape.0.extend_from_slice(shape.get().as_slice());
        } else {
            eprintln!("Tensor already has rank: {}", self.get_shape());
            exit(ExitCode::AstError);
        }
    }

    pub fn transpose(&self) -> Self {
        TypeTensor::new(&self.get_shape().transpose(), self.t)
    }
}

////////////////////////////////////////
//  Expr implementation section
////////////////////////////////////////

impl Binop {
    pub fn from_str(op: &str) -> Self {
        match op {
            "+"     => Binop::Add,
            "/"     => Binop::Div,
            ".*"    => Binop::MatMul,
            "*"     => Binop::Mul,
            "-"     => Binop::Sub,
            _   => {
                eprintln!("Unexpected op string '{}'", op);
                exit(ExitCode::AstError);
            },
        }
    }
}

impl BinopExpr {
    pub fn new(op: Binop, lhs: SharedValue, rhs: SharedValue) -> Self {
        BinopExpr{op, lhs, rhs}
    }

    pub fn get_lhs(&self) -> &SharedValue {
        &self.lhs
    }

    pub fn get_op(&self) -> Binop {
        self.op
    }

    #[allow(dead_code)]
    pub fn get_precedence(&self) -> BinopPrecedence {
        BinopPrecedence::new(self.op)
    }

    pub fn get_rhs(&self) -> &SharedValue {
        &self.rhs
    }

    pub fn get_symbol(&self) -> Symbol {
        Symbol::new_function(match self.get_op() {
            Binop::Add      => "add",
            Binop::Div      => "div",
            Binop::MatMul   => "matmul",
            Binop::Mul      => "mul",
            Binop::Sub      => "sub",
        })
    }
}

impl BinopPrecedence {
    pub fn new(op: Binop) -> Self {
        BinopPrecedence{0: match op {
            Binop::Add      => 20,
            Binop::Div      => 40,
            Binop::MatMul   => 80,
            Binop::Mul      => 40,
            Binop::Sub      => 20,
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
    pub fn new(name: String, args: &SharedValues) -> Self {
        CallExpr{name, args: args.clone()}
    }

    pub fn get_args(&self) -> &SharedValues {
        &self.args
    }

    pub fn get_callee(&self) -> &String {
        &self.name
    }

    pub fn get_symbol(&self) -> Symbol {
        Symbol::new_function(self.get_callee())
    }
}

impl FunctionExpr {
    pub fn new(proto: SharedValue, values: &SharedValues) -> Self {
        FunctionExpr{proto, values: values.clone()}
    }

    pub fn get_body(&self) -> &SharedValues {
        &self.values
    }

    pub fn get_prototype(&self) -> &SharedValue {
        &self.proto
    }

    pub fn get_name(&self) -> &String {
        self.proto.get_kind().to_prototype().unwrap().get_name()
    }

    pub fn get_symbol(&self) -> Symbol {
        self.proto.get_kind().to_prototype().unwrap().get_symbol()
    }
}

impl LiteralExpr {
    pub fn new(shape: &Shape, values: &SharedValues) -> Self {
        LiteralExpr{shape: shape.clone(), values: values.clone()}
    }

    pub fn get_shape(&self) -> &Shape {
        &self.shape
    }

    pub fn get_values(&self) -> &SharedValues {
        &self.values
    }
}

impl ModuleExpr {
    pub fn new(name: String, values: &SharedValues) -> Self {
        ModuleExpr{name, values: values.clone()}
    }

    pub fn get_functions(&self) -> &SharedValues {
        &self.values
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn get_symbol(&self) -> Symbol {
        Symbol::new(self.get_name())
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
    pub fn new(value: SharedValue) -> Self {
        PrintExpr{value}
    }

    pub fn get_symbol(&self) -> Symbol {
        Symbol::new_function("print")
    }

    pub fn get_value(&self) -> &SharedValue {
        &self.value
    }
}

impl PrototypeExpr {
    pub fn new(name: String, values: &SharedValues) -> Self {
        PrototypeExpr{name, values: values.clone()}
    }

    pub fn get_args(&self) -> &SharedValues {
        &self.values
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn get_symbol(&self) -> Symbol {
        Symbol::new_function(self.get_name())
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl ReturnExpr {
    pub fn new(value: Option<SharedValue>) -> Self {
        ReturnExpr{value}
    }

    pub fn get_value(&self) -> &Option<SharedValue> {
        &self.value
    }

    pub fn is_empty(&self) -> bool {
        self.value.is_none()
    }
}

impl TransposeExpr {
    pub fn new(value: SharedValue) -> Self {
        TransposeExpr{value}
    }

    pub fn get_symbol(&self) -> Symbol {
        Symbol::new_function("transpose")
    }

    pub fn get_value(&self) -> &SharedValue {
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

impl SharedValues {
    pub fn new(values: &Vec<SharedValue>) -> Self {
        SharedValues{0: values.clone()}
    }

    pub fn get(&self, i: usize) -> &SharedValue {
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

    pub fn iter(&self) -> slice::Iter<SharedValue> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn push(&mut self, value: SharedValue) -> () {
        self.0.push(value)
    }
}

impl VarExpr {
    pub fn new(name: String, is_param: bool) -> Self {
        VarExpr{name, is_param}
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn get_symbol(&self) -> Symbol {
        Symbol::new(self.get_name())
    }

    pub fn is_param(&self) -> bool {
        self.is_param
    }
}

impl VarDeclExpr {
    pub fn new(name: String, shape: Shape, value: SharedValue) -> Self {
        VarDeclExpr{name, shape, value}
    }

    pub fn get_name(&self) -> &String {
        &self.name
    }

    pub fn get_shape(&self) -> &Shape {
        &self.shape
    }

    pub fn get_symbol(&self) -> Symbol {
        Symbol::new(self.get_name())
    }

    pub fn get_value(&self) -> &SharedValue {
        &self.value
    }

    pub fn is_shapeless(&self) -> bool {
        self.shape.0.is_empty()
    }
}

impl Expr {
    pub fn as_shared(&self) -> SharedValue {
        Rc::from(self.clone())
    }

    #[allow(dead_code)]
    pub fn as_value(&mut self) -> Value {
        Value::new(mem::take(self))
    }

    pub fn get_kind(&self) -> &ExprKind {
        &self.kind
    }

    pub fn get_kind_id(&self) -> ExprKindID {
        self.id
    }

    pub fn is(&self, id: ExprKindID) -> bool {
        self.get_kind_id() == id
    }

    #[allow(dead_code)]
    pub fn is_one_of(&self, ks: &[ExprKindID]) -> bool {
        fn f(t: &Expr, acc: bool, _ks: &[ExprKindID]) -> bool {
            match _ks {
                []              => acc,
                [k]             => acc || t.is(*k),
                [k, tail @ ..]  => f(t, acc || t.is(*k), tail),
            }
        }
        f(self, false, ks)
    }

    pub fn new(kind: ExprKind, id: ExprKindID, loc: Location) -> Self {
        Expr{kind, id, loc}
    }

    /// Convenience initializer for ExprKind::Binop
    pub fn new_binop(op: Binop, lhs: SharedValue, rhs: SharedValue, loc: Location) -> Self {
        Expr::new(ExprKind::Binop(BinopExpr::new(op, lhs, rhs)), ExprKindID::Binop, loc)
    }

    /// Convenience initializer for ExprKind::Call
    pub fn new_call(name: String, args: &SharedValues, loc: Location) -> Self {
        Expr::new(ExprKind::Call(CallExpr::new(name, args)), ExprKindID::Call, loc)
    }

    /// Convenience initializer for ExprKind::Function
    pub fn new_function(proto: SharedValue, values: &SharedValues, loc: Location) -> Self {
        Expr::new(ExprKind::Function(FunctionExpr::new(proto, values)), ExprKindID::Function, loc)
    }

    /// Convenience initializer for ExprKind::Literal
    pub fn new_literal(shape: &Shape, values: &SharedValues, loc: Location) -> Self {
        Expr::new(ExprKind::Literal(LiteralExpr::new(shape, values)), ExprKindID::Literal, loc)
    }

    /// Convenience initializer for ExprKind::Module
    pub fn new_module(name: String, values: &SharedValues, loc: Location) -> Self {
        Expr::new(ExprKind::Module(ModuleExpr::new(name, values)), ExprKindID::Module, loc)
    }

    /// Convenience initializer for ExprKind::Number
    pub fn new_number(value: TMlp, loc: Location) -> Self {
        Expr::new(ExprKind::Number(NumberExpr::new(value)), ExprKindID::Number, loc)
    }

    /// Convenience initializer for ExprKind::Var which is a parameter
    pub fn new_param(name: String, loc: Location) ->  Self {
        Expr::new(ExprKind::Var(VarExpr::new(name, true)), ExprKindID::Var, loc)
    }

    /// Convenience initializer for ExprKind::Print
    pub fn new_print(value: SharedValue, loc: Location) -> Self {
        Expr::new(ExprKind::Print(PrintExpr::new(value)), ExprKindID::Print, loc)
    }

    /// Convenience initializer for ExprKind::Prototype
    pub fn new_prototype(name: String, values: &SharedValues, loc: Location) -> Self {
        Expr::new(ExprKind::Prototype(PrototypeExpr::new(name, values)), ExprKindID::Prototype, loc)
    }

    /// Convenience initializer for ExprKind::Return
    pub fn new_return(value: Option<SharedValue>, loc: Location) -> Self {
        Expr::new(ExprKind::Return(ReturnExpr::new(value)), ExprKindID::Return, loc)
    }

    /// Convenience initializer for ExprKind::Transpose
    pub fn new_transpose(value: SharedValue, loc: Location) -> Self {
        Expr::new(ExprKind::Transpose(TransposeExpr::new(value)), ExprKindID::Transpose, loc)
    }

    /// Convenience initializer for ExprKind::Var
    pub fn new_var(name: String, loc: Location) ->  Self {
        Expr::new(ExprKind::Var(VarExpr::new(name, false)), ExprKindID::Var, loc)
    }

    /// Convenience initializer for ExprKind::VarDecl
    pub fn new_var_decl(name: String, shape: Shape, value: SharedValue, loc: Location) -> Self {
        Expr::new(ExprKind::VarDecl(VarDeclExpr::new(name, shape, value)), ExprKindID::VarDecl, loc)
    }
}

impl Default for Expr {
    fn default() -> Self {
        Expr::new(ExprKind::Unset(), ExprKindID::Unset, Default::default())
    }
}

impl ExprKind {
    pub fn get_symbol(&self) -> Option<Symbol> {
        match self {
            ExprKind::Binop(expr)       => Some(expr.get_symbol()),
            ExprKind::Call(expr)        => Some(expr.get_symbol()),
            ExprKind::Function(expr)    => Some(expr.get_symbol()),
            ExprKind::Literal(_expr)    => None,
            ExprKind::Module(expr)      => Some(expr.get_symbol()),
            ExprKind::Number(_expr)     => None,
            ExprKind::Print(expr)       => Some(expr.get_symbol()),
            ExprKind::Prototype(expr)   => Some(expr.get_symbol()),
            ExprKind::Return(_expr)     => None,
            ExprKind::Transpose(expr)   => Some(expr.get_symbol()),
            ExprKind::Unset()           => None,
            ExprKind::Var(expr)         => Some(expr.get_symbol()),
            ExprKind::VarDecl(expr)     => Some(expr.get_symbol()),
        }
    }

    pub fn to_binop(&self) -> Option<&BinopExpr> {
        match self {
            ExprKind::Binop(expr)   => Some(expr),
            _                       => None,
        }
    }

    pub fn to_call(&self) -> Option<&CallExpr> {
        match self {
            ExprKind::Call(expr)    => Some(expr),
            _                       => None,
        }
    }

    pub fn to_function(&self) -> Option<&FunctionExpr> {
        match self {
            ExprKind::Function(expr)    => Some(expr),
            _                           => None,
        }
    }

    pub fn to_literal(&self) -> Option<&LiteralExpr> {
        match self {
            ExprKind::Literal(expr) => Some(expr),
            _                       => None,
        }
    }

    pub fn to_module(&self) -> Option<&ModuleExpr> {
        match self {
            ExprKind::Module(expr)  => Some(expr),
            _                       => None,
        }
    }

    pub fn to_number(&self) -> Option<&NumberExpr> {
        match self {
            ExprKind::Number(expr)  => Some(expr),
            _                       => None,
        }
    }

    pub fn to_print(&self) -> Option<&PrintExpr> {
        match self {
            ExprKind::Print(expr)   => Some(expr),
            _                       => None,
        }
    }

    pub fn to_prototype(&self) -> Option<&PrototypeExpr> {
        match self {
            ExprKind::Prototype(expr)   => Some(expr),
            _                           => None,
        }
    }

    pub fn to_return(&self) -> Option<&ReturnExpr> {
        match self {
            ExprKind::Return(expr)  => Some(expr),
            _                       => None,
        }
    }

    pub fn to_transpose(&self) -> Option<&TransposeExpr> {
        match self {
            ExprKind::Transpose(expr)   => Some(expr),
            _                           => None,
        }
    }

    pub fn to_var(&self) -> Option<&VarExpr> {
        match self {
            ExprKind::Var(expr) => Some(expr),
            _                   => None,
        }
    }

    pub fn to_var_decl(&self) -> Option<&VarDeclExpr> {
        match self {
            ExprKind::VarDecl(expr) => Some(expr),
            _                       => None,
        }
    }
}

////////////////////////////////////////
//  Display implementation section
////////////////////////////////////////

impl fmt::Display for Binop {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Binop::Add      => "+",
            Binop::Div      => "/",
            Binop::MatMul   => ".*",
            Binop::Mul      => "*",
            Binop::Sub      => "-",
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
        write!(f, "{}Call '{}': {}", indent, self.name, loc)?;
        if !self.args.is_empty() {
            write!(f, "\n")?;
            self.args.expr_fmt(f, depth + 1, loc)?;
        }
        Ok(())
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
        for (i, value) in self.values.0.iter().enumerate() {
            match value.get_kind() {
                ExprKind::Var(expr) => write!(f, "{}", expr.get_name())?,
                _                   => {
                    eprintln!("Expected variable expression for parameter: {}", value);
                    exit(ExitCode::AstError);
                },
            };
            if i < self.values.0.len() - 1 {
                write!(f, ",")?;
            }
        }
        write!(f, "]")
    }
}

impl ExprDisplay for ReturnExpr {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, loc: &Location) -> fmt::Result {
        let indent = Self::indent(depth);
        if self.is_empty() {
            write!(f, "{}Return: {}\n", indent, loc)
        } else {
            write!(f, "{}Return: {}\n", indent, loc)?;
            let value = self.value.as_ref().unwrap();
            value.get_kind().expr_fmt(f, depth + 1, value.get_loc())
        }
    }
}

impl ExprDisplay for SharedValues {
    fn expr_fmt(&self, f: &mut fmt::Formatter, depth: usize, _loc: &Location) -> fmt::Result {
        if !self.is_empty() {
            let indent = Self::indent(depth);
            let is_num = self.0.get(0).unwrap().is(ExprKindID::Number);
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

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            Type::Undef         => "undef".to_string(),
            Type::Unit          => "()".to_string(),
            Type::Scalar(t)     => t.to_string(),
            Type::Sig(t_box)    => t_box.to_string(),
            Type::Tensor(t)     => t.to_string(),
        })
    }
}

impl fmt::Display for TypeBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            TypeBase::F64 => "f64",
        })
    }
}

impl fmt::Display for TypeSignature {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let n = self.get_params().len();
        write!(f, "fn(")?;
        for (i, t) in self.get_params().iter().enumerate() {
            write!(f, "{}", t)?;
            if i < n - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, " -> {})", self.get_return())
    }
}

impl fmt::Display for TypeTensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "<")?;
        if self.is_unranked() {
            write!(f, "*")?;
        } else {
            let n = self.get_shape().get().len();
            for (i, dim) in self.get_shape().get().iter().enumerate() {
                write!(f, "{}", dim)?;
                if i < n - 1 {
                    write!(f, "x")?;
                }
            }
        }
        write!(f, "x{}>", self.get_type())
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
            let is_num = self.0.get(0).unwrap().is(ExprKindID::Number);
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
        write!(f, "{}VarDecl: {}{} {}\n", indent, self.name, self.shape, loc)?;
        self.value.get_kind().expr_fmt(f, depth + 1, self.value.get_loc())
    }
}
