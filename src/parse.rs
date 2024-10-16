// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

use std::str::FromStr;

use crate::ast;
use crate::exit_code;
use crate::lex;
use crate::options;

use ast::Ast;
use ast::Binop;
use ast::BinopExpr;
use ast::BinopPrecedence;
use ast::CallExpr;
use ast::Dims;
use ast::ExprKind;
use ast::ExprKindID;
use ast::Expr;
use ast::FunctionExpr;
use ast::LiteralExpr;
use ast::Location;
use ast::ModuleExpr;
use ast::NumberExpr;
use ast::PrintExpr;
use ast::PrototypeExpr;
use ast::ReturnExpr;
use ast::SharedValue;
use ast::SharedValues;
use ast::Shape;
use ast::TDim;
use ast::TMlp;
use ast::TransposeExpr;
use ast::Value;
use ast::Values;
use ast::VarExpr;
use ast::VarDeclExpr;
use exit_code::exit;
use exit_code::ExitCode;
use lex::Token;
use lex::TokenKind;
use options::RunOptions;
use options::VerboseMode;

const BINOPS: [TokenKind; 5] = [
    TokenKind::DotStar, TokenKind::Minus, TokenKind::Plus, TokenKind::Slash, TokenKind::Star
];

const BUILTINS: [TokenKind; 2] = [
    TokenKind::Print, TokenKind::Transpose
];

const BUILTINS_AND_IDENTS: [TokenKind; 3] = [
    TokenKind::Print, TokenKind::Transpose, TokenKind::Ident
];

const RESERVED_FUNCTIONS: [&str; 5] = [
    "add", "div", "matmul", "mul", "sub"
];

#[derive(Clone)]
pub struct ParserIter {
    token: Token,
    vars: Vec<String>,
    position: usize,
    end: usize,
}

impl ParserIter {
    pub fn new(end: usize) -> Self {
        ParserIter{
            token: Default::default(),
            vars: Vec::new(),
            position: 0,
            end,
        }
    }

    pub fn has_next(&self) -> bool {
        self.position < self.end
    }
}

pub struct Parser<'a> {
    tokens: &'a Vec<Token>,
    options: &'a RunOptions,
}

impl <'a> Parser<'a> {
    pub fn new(tokens: &'a Vec<Token>, options: &'a RunOptions) -> Self {
        if tokens.is_empty() {
            eprintln!("Found empty program while parsing");
            exit(ExitCode::ParserError);
        }
        Parser{tokens, options}
    }

    pub fn iter(&'a self) -> ParserIter {
        ParserIter::new(self.tokens.len())
    }

    /// Does not advance the iterator based on the token result
    fn check(&'a self, iter: &mut ParserIter, k: TokenKind) -> bool {
        let t: &Token = self.get_token(iter);
        if t.is(k) {
            if self.options.is_verbose(VerboseMode::Parser) {
                eprintln!("Found expected token '{}' at position '{}'", k, iter.position);
            }
            true
        } else {
            false
        }
    }

    fn check_one_of(&'a self, iter: &mut ParserIter, ks: &[TokenKind]) -> bool {
        for k in ks {
            if self.check(iter, *k) { return true }
        }
        false
    }

    /// Advances the iterator if token is correctly guessed; else false
    fn consume(&'a self, iter: &mut ParserIter, k: TokenKind, add_var: bool) -> bool {
        let t: &Token = self.get_token(iter);
        if t.is(k) {
            if self.options.is_verbose(VerboseMode::Parser) {
                eprintln!("Consumed expected token '{}' at position '{}'", t, iter.position);
            }
            iter.token = t.clone();
            iter.position += 1;
            if add_var && t.is(TokenKind::Ident) {
                iter.vars.push(t.text.clone());
            }
            true
        } else {
            false
        }
    }

    fn consume_empty_statements(&'a self, iter: &mut ParserIter) -> () {
        while self.consume(iter, TokenKind::Semicolon, false) {}
    }

    fn consume_one_of(&'a self, iter: &mut ParserIter, ks: &[TokenKind], add_var: bool) -> bool {
        for k in ks {
            if self.consume(iter, *k, add_var) { return true }
        }
        false
    }

    /// Advances the iterator if token is correctly guessed; else error
    fn expect(&'a self, iter: &mut ParserIter, k: TokenKind, add_var: bool) -> () {
        if !self.consume(iter, k, add_var) {
            eprintln!("Expected '{}' token at position '{}'", k, iter.position);
            exit(ExitCode::ParserError);
        }
    }

    fn expect_one_of(&'a self, iter: &mut ParserIter, ks: &[TokenKind], add_var: bool) -> () {
        for k in ks {
            if self.consume(iter, *k, add_var) { return }
        }
        eprintln!("Expected one of '{:?}' token at position '{}'", ks, iter.position);
        exit(ExitCode::ParserError);
    }

    fn get_location(&'a self, iter: &'a mut ParserIter) -> Location {
        self.get_prev_token(iter).get_loc().clone()
    }

    fn get_prev_token(&'a self, iter: &'a mut ParserIter) -> &Token {
        &iter.token
    }

    fn get_token(&'a self, iter: &mut ParserIter) -> &Token {
        if iter.has_next() {
            self.tokens.get(iter.position).unwrap()
        } else {
            eprintln!("Token out of bounds at {}", iter.position);
            exit(ExitCode::ParserError);
        }
    }

    fn is_hex_number(text: &str) -> bool {
        text.len() >= 2 && "0x" == &text[0..2]
    }

    fn str_to_tdim(text: &String) -> TDim {
        if Self::is_hex_number(text) {
            Self::emit_error(format!("Hexadecimal dimensions are illegal: {}", text));
        } else {
            match i64::from_str(text.as_str()) {
                Ok(n)   => if n > 0 {
                    n as TDim
                } else {
                    Self::emit_error(format!("Dimensions must be positive integers: {}", text));
                },
                Err(e)  => Self::emit_error(
                    format!("Failed to convert dimension string '{}': {}", text, e)
                ),
            }
        }
    }

    fn str_to_tmlp(text: &String) -> TMlp {
        if Self::is_hex_number(text) {
            match u64::from_str(text.as_str()) {
                Ok(n)   => f64::from_bits(n),
                Err(e)  => Self::emit_error(
                    format!("Failed to convert hexadecimal string '{}': {}", text, e)
                ),
            }
        } else {
            match f64::from_str(text.as_str()) {
                Ok(n)   => n,
                Err(e)  => Self::emit_error(
                    format!("Failed to convert decimal string '{}': {}", text, e)
                ),
            }
        }
    }

    /// Implement a recursive operatator-precedence parser [1][2].
    /// [1]: https://en.wikipedia.org/wiki/Operator-precedence_parser#Pseudocode
    /// [2]: https://github.com/llvm/llvm-project/blob/main/mlir/examples/toy/Ch1/include/toy/Parser.h#L244
    fn parse_binop_rhs(
        &'a self,
        iter: &mut ParserIter,
        prec: BinopPrecedence,
        value: SharedValue
    ) -> SharedValue {
        let mut lhs = value;
        loop {
            if !self.check_one_of(iter, &BINOPS) {
                return lhs;
            }
            let op = self.get_token(iter).text.clone();
            let binop = Binop::from_str(&op);
            let prec_op = BinopPrecedence::new(binop);
            if prec_op < prec {
                return lhs;
            }
            self.expect_one_of(iter, &BINOPS, false);
            let loc = self.get_location(iter).clone();
            let mut rhs = match self.parse_primary(iter) {
                Some(value) => value,
                None        => Self::emit_error(format!("Expected expression after '{}'", op)),
            };
            if self.check_one_of(iter, &BINOPS) {
                let op_next = self.get_token(iter).text.clone();
                let binop_next = Binop::from_str(&op_next);
                if prec_op < BinopPrecedence::new(binop_next) {
                    rhs = self.parse_binop_rhs(iter, prec_op.next(), rhs);
                }
            }
            lhs = Expr::new_binop(binop, lhs, rhs, loc.clone()).as_shared();
        }
    }

    fn parse_block(&'a self, iter: &mut ParserIter, values: &mut SharedValues) -> () {
        if !self.consume(iter, TokenKind::BraceL, false) {
            Self::emit_error("Expected '{' token".to_string());
        }
        self.consume_empty_statements(iter);
        while !self.check(iter, TokenKind::BraceR) {
            if self.check_one_of(iter, &[TokenKind::Eoi, TokenKind::Return, TokenKind::Var]) {
                let token = self.get_token(iter);
                match token.kind {
                    TokenKind::Eoi      => break,
                    TokenKind::Return   => values.push(self.parse_return_statement(iter)),
                    TokenKind::Var      => values.push(self.parse_declaration(iter)),
                    _                   => Self::emit_error(
                        format!("Unexpected token '{}'", token.text)
                    ),
                };
            } else {
                match self.parse_expression(iter) {
                    Some(value) => values.push(value),
                    None        => (),
                };
            }
            self.expect(iter, TokenKind::Semicolon, false);
            self.consume_empty_statements(iter);
        }
        self.expect(iter, TokenKind::BraceR, false);
    }

    fn parse_call(&'a self, iter: &mut ParserIter, name: String) -> SharedValue {
        Self::check_reserved(name.as_str());
        let loc = self.get_location(iter);
        match name.as_str() {
            "print"     => {
                let args = match self.parse_paren_expr(iter) {
                    Some(v) => v,
                    None    => Self::emit_error("Expected 1 arg to print call".to_string()),
                };
                Expr::new_print(args, loc)
            },
            "transpose" => {
                let arg = match self.parse_paren_expr(iter) {
                    Some(v) => v,
                    None    => Self::emit_error("Expected 1 arg to transpose call".to_string()),
                };
                Expr::new_transpose(arg, loc)
            },
            _           => {
                let mut args: SharedValues = Default::default();
                self.parse_paren_expr_list(iter, &mut args);
                Expr::new_call(name.clone(), &args, loc)
            },
        }.as_shared()
    }

    fn parse_declaration(&'a self, iter: &mut ParserIter) -> SharedValue {
        self.expect(iter, TokenKind::Var, false);
        let loc = self.get_location(iter);
        self.expect(iter, TokenKind::Ident, true);
        let name = self.get_prev_token(iter).text.clone();
        let shape = if self.check(iter, TokenKind::AngleL) {
            self.parse_type(iter)
        } else {
            Default::default()
        };
        self.expect(iter, TokenKind::Assign, false);
        match self.parse_expression(iter) {
            Some(value) => Expr::new_var_decl(name, shape, value, loc).as_shared(),
            None        => Self::emit_error("Expected initialization for declaration".to_string()),
        }
    }

    /// Optimize empty function definitions out
    fn parse_definition(&'a self, iter: &mut ParserIter) -> Option<SharedValue> {
        let proto = self.parse_prototype(iter);
        let loc = proto.get_loc().clone();
        let mut values: SharedValues = Default::default();
        self.parse_block(iter, &mut values);
        if !values.is_empty() {
            Some(Expr::new_function(proto, &values, loc).as_shared())
        } else {
            None
        }
    }

    fn parse_expression(&'a self, iter: &mut ParserIter) -> Option<SharedValue> {
        let value = match self.parse_primary(iter) {
            Some(v) => v,
            None    => return None,
        };
        Some(self.parse_binop_rhs(iter, Default::default(), value))
    }

    fn parse_ident_expr(&'a self, iter: &mut ParserIter) -> SharedValue {
        self.expect_one_of(iter, &BUILTINS_AND_IDENTS, false);
        let name = self.get_prev_token(iter).text.clone();
        let loc = self.get_location(iter);
        if self.get_prev_token(iter).is_one_of(&BUILTINS) {
            self.parse_call(iter, name)
        } else {
            if self.check(iter, TokenKind::ParenL) {
                self.parse_call(iter, name)
            } else {
                Expr::new_var(name, loc).as_shared()
            }
        }
    }

    fn parse_literal_list(&'a self, iter: &mut ParserIter, values: &mut SharedValues) -> Location {
        self.expect(iter, TokenKind::BracketL, false);
        let loc = self.get_location(iter);
        loop {
            if self.check(iter, TokenKind::BracketL) {
                values.push(self.parse_tensor_literal(iter));
            } else {
                values.push(self.parse_number_expr(iter));
            }
            if !self.consume(iter, TokenKind::Comma, false) {
                break;
            }
        }
        self.expect(iter, TokenKind::BracketR, false);
        if values.is_empty() {
            Self::emit_error("Unexpected empty literal expression".to_string());
        }
        loc
    }

    fn parse_module(&'a self, name: &str, iter: &mut ParserIter) -> SharedValue {
        let mut functions: SharedValues = Default::default();
        while !self.consume(iter, TokenKind::Eoi, false) {
            match self.parse_definition(iter) {
                Some(value) => functions.push(value),
                None        => (),
            }
        }
        if functions.is_empty() {
            Self::emit_error(format!("Unexpected empty module '{}'", name));
        }
        Expr::new_module(name.to_string(), &functions, self.get_location(iter)).as_shared()
    }

    fn parse_number_expr(&'a self, iter: &mut ParserIter) -> SharedValue {
        self.expect(iter, TokenKind::Number, false);
        let n: TMlp = Self::str_to_tmlp(&self.get_prev_token(iter).text);
        Expr::new_number(n, self.get_location(iter)).as_shared()
    }

    fn parse_param_list(&'a self, iter: &mut ParserIter, values: &mut SharedValues) -> () {
        self.expect(iter, TokenKind::ParenL, false);
        if !self.check(iter, TokenKind::ParenR) {
            loop {
                self.expect(iter, TokenKind::Ident, false);
                let loc = self.get_location(iter);
                let name = &self.get_prev_token(iter).text;
                values.push(Expr::new_param(name.clone(), loc.clone()).as_shared());
                if !self.consume(iter, TokenKind::Comma, false) {
                    break;
                }
            }
        }
        self.expect(iter, TokenKind::ParenR, false);
    }

    fn parse_paren_expr(&'a self, iter: &mut ParserIter) -> Option<SharedValue> {
        self.expect(iter, TokenKind::ParenL, false);
        let value = self.parse_expression(iter);
        self.expect(iter, TokenKind::ParenR, false);
        value
    }

    fn parse_paren_expr_list(&'a self, iter: &mut ParserIter, values: &mut SharedValues) -> () {
        self.expect(iter, TokenKind::ParenL, false);
        loop {
            if self.check(iter, TokenKind::ParenR) {
                break;
            }
            match self.parse_expression(iter) {
                Some(value) => values.push(value),
                None        => break,
            };
            if !self.consume(iter, TokenKind::Comma, false) {
                break;
            }
        }
        self.expect(iter, TokenKind::ParenR, false);
    }

    fn parse_primary(&'a self, iter: &mut ParserIter) -> Option<SharedValue> {
        if self.check_one_of(iter, &BUILTINS_AND_IDENTS) {
            Some(self.parse_ident_expr(iter))
        } else if self.check(iter, TokenKind::Minus) {
            // TODO: Should be typed to rhs if more types added
            let value_zero = Expr::new_number(TMlp::default(), self.get_location(iter)).as_shared();
            Some(self.parse_binop_rhs(iter, Default::default(), value_zero))
        } else if self.check(iter, TokenKind::Number) {
            Some(self.parse_number_expr(iter))
        } else if self.check(iter, TokenKind::ParenL) {
            self.parse_paren_expr(iter)
        } else if self.check(iter, TokenKind::BracketL) {
            Some(self.parse_tensor_literal(iter))
        } else if self.consume_one_of(iter, &[TokenKind::Semicolon, TokenKind::BraceR], false) {
            None
        } else {
            Self::emit_error(format!("Unexpected token '{}'", self.get_token(iter).text));
        }
    }

    fn parse_prototype(&'a self, iter: &mut ParserIter) -> SharedValue {
        self.expect(iter, TokenKind::Def, false);
        self.expect(iter, TokenKind::Ident, true);
        let loc = self.get_location(iter);
        let name = self.get_prev_token(iter).text.clone();
        Self::check_reserved(name.as_str());
        let mut args: SharedValues = Default::default();
        self.parse_param_list(iter, &mut args);
        Expr::new_prototype(name, &args, loc).as_shared()
    }

    fn parse_return_statement(&'a self, iter: &mut ParserIter) -> SharedValue {
        self.expect(iter, TokenKind::Return, false);
        let loc = self.get_location(iter);
        Expr::new_return(
            if self.check(iter, TokenKind::Semicolon) {
                None
            } else {
                self.parse_expression(iter)
            },
            loc
        ).as_shared()
    }

    fn parse_tensor_literal(&'a self, iter: &mut ParserIter) -> SharedValue {
        let mut dims = Dims::new();
        let mut values: SharedValues = Default::default();
        let loc = self.parse_literal_list(iter, &mut values);
        dims.push(values.len() as TDim);
        let mut is_nested = false;
        let mut shape_first = Default::default();
        for (i, value) in values.iter().enumerate() {
            if value.is(ExprKindID::Literal) {
                let expr = match value.get_kind() {
                    ExprKind::Literal(expr) => expr,
                    _                       => Self::emit_error("Unexpected expression".to_string()),
                };
                let shape = expr.get_shape();
                if !is_nested {
                    is_nested = true;
                    shape_first = shape.clone();
                } else if shape_first != *shape {
                    Self::emit_error(
                        format!("Unexpected non-uniform nested literal expression: {}", value)
                    );
                }
                if i == 0 {
                    dims.append(&mut shape.get().clone());
                }
            } else if value.is(ExprKindID::Number) {
                if is_nested {
                    Self::emit_error(
                        format!("Unexpected non-uniform nested literal expression: {}", value)
                    );
                }
            } else {
                Self::emit_error(
                    format!("Unexpected value in literal expression: {}", value)
                );
            }
        }
        Expr::new_literal(&Shape::new(&dims), &values, loc).as_shared()
    }

    fn parse_type(&'a self, iter: &mut ParserIter) -> Shape {
        self.expect(iter, TokenKind::AngleL, false);
        let mut dims: Dims = Dims::new();
        while self.consume(iter, TokenKind::Number, false) {
            let text = &self.get_prev_token(iter).text;
            dims.push(Self::str_to_tdim(text));
            if !self.consume(iter, TokenKind::Comma, false) {
                break;
            }
        }
        self.expect(iter, TokenKind::AngleR, false);
        Shape::new(&dims)
    }

    pub fn parse_input(
        ret: &mut SharedValue,
        parser: &'a mut Parser<'a>,
        name: &str,
        options: &RunOptions
    ) {
        let mut iter = parser.iter();
        *ret = parser.parse_module(name, &mut iter);
        if options.print_ast { eprintln!("AST:\n{}", ret); }
        if options.parse_exit { exit(ExitCode::Ok); }
    }

    fn check_reserved(s: &str) -> () {
        if RESERVED_FUNCTIONS.contains(&s) {
            Self::emit_error(format!("Reserved function name '{}'", s));
        }
    }

    fn emit_error(message: String) -> ! {
        eprintln!("{}", message);
        exit(ExitCode::ParserError);
    }
}
