// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

use std::fmt;
use std::fmt::Display;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;

use crate::ast;
use crate::exit_code;
use crate::options;

use ast::Location;
use exit_code::exit;
use exit_code::ExitCode;
use options::RunOptions;
use options::VerboseMode;

#[derive(Clone,Copy,Debug,Default,Eq,PartialEq)]
pub enum TokenKind {
    #[default]
    Unknown,
    AngleL,
    AngleR,
    Assign,
    BraceL,
    BraceR,
    BracketL,
    BracketR,
    Comma,
    Comment,
    Colon,
    Def,
    DotStar,
    Eoi,
    Eol,
    Ident,
    Minus,
    Number,
    ParenL,
    ParenR,
    Print,
    Plus,
    Return,
    Semicolon,
    Slash,
    Star,
    Transpose,
    Var,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            TokenKind::AngleL       => "AngleL",
            TokenKind::AngleR       => "AngleR",
            TokenKind::Assign       => "Assign",
            TokenKind::BraceL       => "BraceL",
            TokenKind::BraceR       => "BraceR",
            TokenKind::BracketL     => "BracketL",
            TokenKind::BracketR     => "BracketR",
            TokenKind::Comma        => "Comma",
            TokenKind::Comment      => "Comment",
            TokenKind::Colon        => "Colon",
            TokenKind::Def          => "Def",
            TokenKind::DotStar      => "DotStar",
            TokenKind::Eoi          => "Eoi",
            TokenKind::Eol          => "Eol",
            TokenKind::Ident        => "Ident",
            TokenKind::Minus        => "Minus",
            TokenKind::Number       => "Number",
            TokenKind::ParenL       => "ParenL",
            TokenKind::ParenR       => "ParenR",
            TokenKind::Print        => "Print",
            TokenKind::Plus         => "Plus",
            TokenKind::Return       => "Return",
            TokenKind::Semicolon    => "Semicolon",
            TokenKind::Slash        => "Slash",
            TokenKind::Star         => "Star",
            TokenKind::Transpose    => "Transpose",
            TokenKind::Unknown      => "Unknown",
            TokenKind::Var          => "Var",
        })
    }
}

#[derive(Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub text: String,
    pub loc: Location,
}

impl Default for Token {
    fn default() -> Self {
        Token::new(TokenKind::Unknown, Default::default(), Default::default())
    }
}

impl Token {
    pub fn new(k: TokenKind, text: String, loc: Location) -> Self {
        Token{kind: k, text, loc}
    }

    pub fn get_loc(&self) -> &Location {
        &self.loc
    }

    pub fn is(&self, k: TokenKind) -> bool {
        self.kind == k
    }

    #[allow(dead_code)]
    pub fn is_one_of(&self, ks: &[TokenKind]) -> bool {
        fn f(t: &Token, acc: bool, _ks: &[TokenKind]) -> bool {
            match _ks {
                []              => acc,
                [k]             => acc || t.is(*k),
                [k, tail @ ..]  => f(t, acc || t.is(*k), tail),
            }
        }
        f(self, false, ks)
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.kind, self.text)
    }
}

pub struct Lexer<'a, T: Read> {
    buffer: BufReader<T>,
    line: String,
    line_count: usize,
    name: String,
    position: usize,
    options: &'a RunOptions,
}

impl <'a, T: Read> Lexer<'a, T> {
    pub fn new(input_name: &str, readable: T, options: &'a RunOptions) -> Self {
        Lexer{
            buffer: BufReader::new(readable),
            line: String::new(),
            line_count: 0,
            name: input_name.to_string(),
            position: 0,
            options,
        }
    }

    fn has_next(&mut self) -> bool {
        if self.has_next_in_line(self.position) { 
            true
        } else {
            if !self.line.is_empty() && self.position >= self.line.len() {
                self.line = Default::default();
            }
            match self.buffer.read_line(&mut self.line) {
                Ok(size) => {
                    if size > 0 {
                        if self.options.is_verbose(VerboseMode::Lexer) {
                            eprintln!("Read {} bytes from buffer at line {}", size, self.line_count);
                        }
                        self.line_count += 1;
                        self.position = 0;
                        true
                    } else {
                        false
                    }
                }
                Err(_) => false,
            }
        }
    }

    fn has_next_in_line(&self, pos: usize) -> bool {
        if self.line.is_empty() {
            false
        } else {
            pos < self.line.len()
        }
    }

    fn next_char_in_line(&self, pos: usize) -> char {
        let c_opt = self.line.get(pos..pos + 1);
        match c_opt {
            None            => {
                eprintln!("Expected char in line {} at pos {}", self.line_count - 1, pos);
                exit(ExitCode::LexerError);
            }
            Some(c_slice)   => {
                let c: char = c_slice.chars().next().unwrap();
                if !c.is_ascii() {
                    eprintln!("Only ASCII characters are supported by the lexer");
                    exit(ExitCode::LexerError);
                }
                if self.options.is_verbose(VerboseMode::Lexer) {
                    eprintln!("Found char '{}' in line {} at pos {}", c, self.line_count - 1, pos);
                }
                c
            }
        }
    }

    fn collect_token_sequence(&self, pos: usize, pred: fn(char) -> bool) -> usize {
        let mut pos_end: usize = pos;
        let mut c: char;
        while self.has_next_in_line(pos_end) {
            c = self.next_char_in_line(pos_end);
            if !pred(c) {
                break
            }
            pos_end += 1;
        }
        pos_end
    }

    fn check_dots(&self, pos_start: usize, pos_end: usize) -> () {
        let mut dots = 0;
        let slice: &str = &self.line[pos_start..pos_end];
        if slice == "." {
            eprintln!("Invalid number '.'");
            exit(ExitCode::LexerError);
        }
        for c in slice.chars() {
            if Self::is_dot(c) {
                dots += 1;
            }
            if dots > 1 {
                eprintln!("Found multiple '.' characters in number '{}'", slice);
                exit(ExitCode::LexerError);
            }
        }
    }

    fn check_suffix(&self, pos: usize) -> () {
        if self.has_next_in_line(pos) {
            let c: char = self.next_char_in_line(pos);
            if !Self::is_whitespace(c) && !Self::is_other(c) {
                eprintln!("Found invalid suffix '{}' for number in expression", c);
                exit(ExitCode::LexerError);
            }
        }
    }

    fn next_in_line(&mut self, t: &mut Token) -> () {
        let (mut c, mut pos_start): (char, usize) = ('\0', self.position);
        while self.has_next_in_line(pos_start) {
            c = self.next_char_in_line(pos_start);
            if !Self::is_whitespace(c) { break }
            pos_start += 1;
        }
        if Self::is_whitespace(c) {
            self.form_token(t, pos_start, pos_start + 1, TokenKind::Eol);
        } else if Self::is_digit(c) || Self::is_dot(c) {
            if Self::is_dot(c) && self.has_next_in_line(pos_start + 1) {
                let c_next = self.next_char_in_line(pos_start + 1);
                if Self::is_star(c_next) {
                    self.form_token(t, pos_start, pos_start + 2, TokenKind::DotStar);
                    return;
                }
            }
            if c == '0' && self.has_next_in_line(pos_start + 1) {
                c = self.next_char_in_line(pos_start + 1);
                if c == 'x' {
                    let pos_end: usize = self.collect_token_sequence(pos_start + 2, Self::is_hex_digit);
                    self.check_suffix(pos_end);
                    self.form_token(t, pos_start, pos_end, TokenKind::Number);
                    return;
                }
            }
            let pos_end: usize = self.collect_token_sequence(pos_start + 1, Self::is_float_digit);
            self.check_suffix(pos_end);
            self.check_dots(pos_start, pos_end);
            self.form_token(t, pos_start, pos_end, TokenKind::Number);
        } else if Self::is_letter(c) {
            let pos_end: usize = self.collect_token_sequence(pos_start + 1, Self::is_ident);
            let text = String::from(&self.line[pos_start..pos_end]);
            self.form_token(
                t,
                pos_start,
                pos_end,
                Self::keyword_token(text.as_str())
            );
        } else if Self::is_slash(c) {
            if self.has_next_in_line(pos_start + 1) {
                c = self.next_char_in_line(pos_start + 1);
                if Self::is_slash(c) {
                    // It's a comment => consume the rest of the line
                    let pos_end: usize = self.collect_token_sequence(pos_start + 2, Self::is_any);
                    self.form_token(t, pos_start, pos_end, TokenKind::Comment);
                    return;
                }
            }
            self.form_token(t, pos_start, pos_start + 1, TokenKind::Slash);
        } else {
            self.form_token(t, pos_start, pos_start + 1, match c {
                '<' => TokenKind::AngleL,
                '>' => TokenKind::AngleR,
                '=' => TokenKind::Assign,
                '{' => TokenKind::BraceL,
                '}' => TokenKind::BraceR,
                '[' => TokenKind::BracketL,
                ']' => TokenKind::BracketR,
                ',' => TokenKind::Comma,
                ':' => TokenKind::Colon,
                '-' => TokenKind::Minus,
                '(' => TokenKind::ParenL,
                ')' => TokenKind::ParenR,
                '+' => TokenKind::Plus,
                ';' => TokenKind::Semicolon,
                '/' => TokenKind::Slash,
                '*' => TokenKind::Star,
                _   => TokenKind::Unknown,
            })
        }
    }

    pub fn next(&mut self, t: &mut Token) -> () {
        let mut t_tmp: Token = Default::default();
        if self.has_next() {
            self.next_in_line(&mut t_tmp);
        } else {
            t_tmp.kind = TokenKind::Eoi;
        }
        std::mem::swap(t, &mut t_tmp);
    }

    fn keyword_token(text: &str) -> TokenKind {
        match text {
            "def"       => TokenKind::Def,
            "print"     => TokenKind::Print,
            "return"    => TokenKind::Return,
            "transpose" => TokenKind::Transpose,
            "var"       => TokenKind::Var,
            _           => TokenKind::Ident,
        }
    }

    fn form_token(&mut self, t: &mut Token, pos_start: usize, pos_end: usize, k: TokenKind) -> () {
        t.kind = k;
        t.text = String::from(
            if k == TokenKind::Eoi || k == TokenKind::Eol {
                ""
            } else {
                &self.line[pos_start..pos_end]
            }
        );
        t.loc = Location::new(self.name.clone(), self.line_count, pos_start + 1);
        self.position = pos_end;
    }

    fn is_any(_c: char) -> bool {
        true
    }

    fn is_dot(c: char) -> bool {
        c == '.'
    }

    fn is_slash(c: char) -> bool {
        c == '/'
    }

    fn is_star(c: char) -> bool {
        c == '*'
    }

    fn is_other(c: char) -> bool {
        matches!(c,
            '<' |
            '>' |
            '{' |
            '}' |
            '[' |
            ']' |
            ';' |
            ',' |
            ':' |
            '-' |
            '(' |
            ')' |
            '+' |
            '/' |
            '*' |
            '='
        )
    }

    fn is_whitespace(c: char) -> bool {
        c == ' ' || c == '\t' || c == '\r' || c == '\n'
    }

    fn is_digit(c: char) -> bool {
        c.is_ascii_digit()
    }

    fn is_float_digit(c: char) -> bool {
        Self::is_digit(c) || Self::is_dot(c)
    }

    fn is_ident(c: char) -> bool {
        Self::is_digit(c) || Self::is_letter(c)
    }

    fn is_hex_digit(c: char) -> bool {
        Self::is_digit(c) || ('a'..='f').contains(&c) || ('A'..='F').contains(&c)
    }

    fn is_letter_lower(c:char) -> bool {
        c.is_ascii_lowercase()
    }

    fn is_letter_upper(c:char) -> bool {
        c.is_ascii_uppercase()
    }

    fn is_letter(c: char) -> bool {
        Self::is_letter_lower(c) || Self::is_letter_upper(c) || c == '_'
    }

    pub fn lex_input(ts: &mut Vec<Token>, lex: &mut Lexer<'a, T>, options: &RunOptions) -> () {
        let mut t: Token = Default::default();
        while !t.is(TokenKind::Eoi) {
            lex.next(&mut t);
            if t.is(TokenKind::Unknown) {
                eprintln!("Found unknown token '{}' in lexer", t.text);
                if !options.drop_token { exit(ExitCode::LexerError); }
            } else if options.is_verbose(VerboseMode::Lexer) {
                eprintln!("Lexed token '{}'", t);
            }
            if t.is(TokenKind::Comment) || t.is(TokenKind::Eol) {
                // Drop the comments and end of lines before parsing
                continue;
            }
            ts.push(t.clone());
        }

        if options.lex_exit { exit(ExitCode::Ok); }
    }
}
