// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause
const LICENSE: &str = "\
// Copyright 2024, Giordano Salvador    \n\
// SPDX-License-Identifier: BSD-3-Clause\n\
";
const PACKAGE: &str = "mlpc";
const VERSION: &str = "v0.1.0";

use std::env;
use std::fmt;
use std::fs::File;
use std::io::Cursor;
use std::io::stdin;
use std::path::Path;

mod ast;
mod command;
mod exit_code;
mod lex;
mod parse;
mod sem;
mod options;

use ast::Ast;
use ast::Expr;
use exit_code::exit;
use exit_code::ExitCode;
use lex::Lexer;
use lex::Token;
use parse::Parser;
use options::BodyType;
use options::CodeGenType;
use options::HostOS;
use options::OptLevel;
use options::OutputType;
use options::RunOptions;
use options::VerboseMode;
use sem::Semantics;

fn help(code: ExitCode) -> ! {
    eprintln!("usage: {} [OPTIONS] <INPUT>\n{}", PACKAGE, [
        "INPUT              '-' (i.e., Stdin) or a file path",
        "OPTIONS:",
        "--ast              Print the AST after parsing",
        "-b|--bitcode       Output LLVM bitcode (post-optimization) (.bc if used with -o)",
        "-B|--bytecode      Output MLIR bytecode (post-optimization) (.mlbc if used with -o)",
        "-c                 Output an object file (post-optimization) (.o if used with -o)",
        "--drop             Drop unknown tokens instead of failing",
        "-e|--expr[=]<E>    Process expression E instead of INPUT file",
        "-h|--help          Print this list of command line options",
        "--lex              Exit after running the lexer",
        "--ir               Exit after printing IR (pre-optimization)",
        "--llvmir           Output LLVM IR (post-optimization) (.ll if used with -o)",
        "-S|--mlir          Output MLIR IR (post-optimization) (.mlir if used with -o)",
        "-k|--no-main       Omit linking with main module (i.e., output kernel only)",
        "                   When this option is selected, an executable cannot be generated",
        "--notarget         Omit target specific configuration in MLIR IR/bytecode",
        "-o[=]<F>           Output to file F instead of Stdout ('-' for Stdout)",
        "                   If no known extension is used (.bc|.exe|.ll|.mlbc|.mlir|.o) an executable is assumed",
        "                   An executable requires llc and clang to be installed",
        "-O<0|1|2|3>        Set the optimization level (default: O2)",
        "--parse            Exit after running the parser",
        "--sem              Exit after running the semantics check",
        "-C|--c-main        Link with a C-derived main module (src/main.c.template)",
        "                   This option is required for generating object files and executables on MacOS",
        "                   and requires clang to be installed",
        "-v|--verbose[[=]M] Enable verbose output (M=[all|lexer|parser|sem|codegen]; default: all)",
        "--version          Display the package version and license information",
    ].join("\n"));
    exit(code);
}

#[derive(Clone,Copy,PartialEq)]
enum InputType<'a> {
    None,
    Stdin,
    Expr(&'a str),
    File(&'a str),
}

impl <'a> fmt::Display for InputType<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            InputType::Stdin    => "Stdin".to_string(),
            InputType::Expr(e)  => format!("Expression:{}", e),
            InputType::File(f)  => format!("File:{}", f),
            InputType::None     => "None".to_string(),
        };
        write!(f, "{}", s)
    }
}

fn set_codegen_type(options: &mut RunOptions, codegen_type: CodeGenType) -> () {
    if options.codegen_type == CodeGenType::Unset {
        options.codegen_type = codegen_type;
    } else {
        eprintln!(
            "Incompatible compiler flags for output type: {} and {}",
            options.codegen_type,
            codegen_type
        );
        exit(ExitCode::ArgParseError);
    }
}

fn set_body_type(options: &mut RunOptions, body_type: BodyType) -> () {
    if options.body_type == BodyType::Unset {
        options.body_type = body_type;
    } else {
        eprintln!("Incompatible compiler flags: '-k|--no-main' and '-C|--c-main'");
        exit(ExitCode::ArgParseError);
    }
}

#[derive(Clone,Copy)]
enum ExtType {
    None,
    Exe,
    BC,
    LL,
    MLBC,
    MLIR,
    O,
}

fn get_extension_from_filename(name: &str) -> ExtType {
    match Path::new(name).extension() {
        None        => ExtType::None,
        Some(ext)   => match ext.to_str().unwrap() {
            "bc"    => ExtType::BC,
            "exe"   => ExtType::Exe,
            "ll"    => ExtType::LL,
            "mlbc"  => ExtType::MLBC,
            "mlir"  => ExtType::MLIR,
            "o"     => ExtType::O,
            _       => ExtType::None,
        },
    }
}

/// Checks to ensure valid combination for BodyType, CodeGenType, and OutputType
fn check_options_configuration(options: &RunOptions, output: &OutputType) -> () {
    match *output {
        OutputType::Stdout  => if !options.early_exit() {
            match options.codegen_type {
                CodeGenType::Object     => {
                    eprintln!("Output to Stdout not supported for object files");
                    exit(ExitCode::ArgParseError);
                },
                CodeGenType::Executable => {
                    eprintln!("Output to Stdout not supported for executable files");
                    exit(ExitCode::ArgParseError);
                },
                _                       => (),
            }
        },
        OutputType::File(f) => {
            let t = options.codegen_type;
            match get_extension_from_filename(f) {
                ExtType::None   => if t != CodeGenType::Executable {
                    eprintln!(
                        "Output name (no/unknown extension) should match codegen type: {} specified",
                        t
                    );
                    exit(ExitCode::ArgParseError);
                },
                ExtType::BC     => if t != CodeGenType::Bitcode {
                    eprintln!("Output name ('.bc' extension) should match codegen type (-b|--bitcode)");
                    exit(ExitCode::ArgParseError);
                },
                ExtType::Exe    => if t != CodeGenType::Executable {
                    eprintln!("Output name ('.exe' extension) should match codegen type: {} specified", t);
                    exit(ExitCode::ArgParseError);
                },
                ExtType::LL     => if t != CodeGenType::Llvmir {
                    eprintln!("Output name ('.ll' extension) should match codegen type (--llvmir)");
                    exit(ExitCode::ArgParseError);
                },
                ExtType::MLBC   => if t != CodeGenType::Bytecode {
                    eprintln!("Output name ('.mlbc' extension) should match codegen type (-B|--bytecode)");
                    exit(ExitCode::ArgParseError);
                },
                ExtType::MLIR   => if t != CodeGenType::Mlir {
                    eprintln!("Output name ('.mlir' extension) should match codegen type (-S|--mlir)");
                    exit(ExitCode::ArgParseError);
                },
                ExtType::O      => if t != CodeGenType::Object {
                    eprintln!("Output name ('.o' extension) should match codegen type (-c)");
                    exit(ExitCode::ArgParseError);
                },
            }
        },
    };

    // If early exit is enabled, don't worry about incompatible output types below
    if options.early_exit() {
        return;
    }

    if options.body_type == BodyType::NoMain && options.codegen_type == CodeGenType::Executable {
        eprintln!("Unsupported -k-|-no-main with executable output type");
        exit(ExitCode::ArgParseError);
    }

    if options.host_os == HostOS::MacOS && options.body_type == BodyType::MainGen {
        eprintln!("Linking the C standard library from a kernel+main module is not supported on MacOS");
        eprintln!("Please use the C-derived main option (-C|--c-main)");
        exit(ExitCode::ArgParseError);
    }
}

fn parse_args<'a>(
    args: &'a [String],
    input: &mut InputType<'a>,
    output: &mut OutputType<'a>,
    options: &mut RunOptions
) {
    let _bin_name: &String = args.first().unwrap();
    let mut arg: &'a String;
    let mut i: usize = 1;

    while i < args.len() {
        arg = args.get(i).unwrap();
        match arg.as_str() {
            "--ast"         => options.print_ast = true,
            "-b"            => set_codegen_type(options, CodeGenType::Bytecode),
            "--bytecode"    => set_codegen_type(options, CodeGenType::Bytecode),
            "-c"            => set_codegen_type(options, CodeGenType::Object),
            "-C"            => set_body_type(options, BodyType::MainGenC),
            "--drop"        => options.drop_token = true,
            "-e"            => *input = InputType::Expr(parse_arg_after(args, &mut i, false).unwrap()),
            "--expr"        => *input = InputType::Expr(parse_arg_after(args, &mut i, false).unwrap()),
            "-h"            => help(ExitCode::Ok),
            "--help"        => help(ExitCode::Ok),
            "--ir"          => options.ir_exit = true,
            "-k"            => set_body_type(options, BodyType::NoMain),
            "--lex"         => options.lex_exit = true,
            "--llvmir"      => set_codegen_type(options, CodeGenType::Llvmir),
            "--mlir"        => set_codegen_type(options, CodeGenType::Mlir),
            "--no-main"     => set_body_type(options, BodyType::NoMain),
            "--notarget"    => options.no_target = true,
            "-o"            => *output = OutputType::new(parse_arg_after(args, &mut i, false).unwrap()),
            "-O0"           => options.opt_level = OptLevel::O0,
            "-O1"           => options.opt_level = OptLevel::O1,
            "-O2"           => options.opt_level = OptLevel::O2,
            "-O3"           => options.opt_level = OptLevel::O3,
            "--parse"       => options.parse_exit = true,
            "--sem"         => options.sem_exit = true,
            "-S"            => set_codegen_type(options, CodeGenType::Mlir),
            "--c-main"      => set_body_type(options, BodyType::MainGenC),
            "-v"            => options.verbose = parse_verbose(parse_arg_after(args, &mut i, true)),
            "--verbose"     => options.verbose = parse_verbose(parse_arg_after(args, &mut i, true)),
            "--version"     => print_pkg_info(true),
            _               => parse_arg_complex(arg, input, output, options),
        }
        i += 1;
    }

    if options.body_type == BodyType::Unset {
        set_body_type(options, BodyType::MainGen);
    }

    if options.codegen_type == CodeGenType::Unset {
        set_codegen_type(options, CodeGenType::Executable);
    }

    if *input == InputType::None {
        eprintln!("No input file/name specified!");
        help(ExitCode::ArgParseError);
    } else if options.is_verbose_any() {
        eprintln!("Processing input '{}'", *input);
    }

    if options.is_verbose_any() {
        eprintln!("Outputting to '{}'", *output);
    }

    check_options_configuration(options, output);
    if options.is_verbose_any() {
        eprintln!("{}", options);
    }
}

fn parse_arg_after<'a>(args: &'a [String], i: &mut usize, none_okay: bool) -> Option<&'a str> {
    let name = args.get(*i).unwrap();
    match args.get(*i + 1) {
        Some(arg) => {
            if arg.len() > 0 {
                let lead_char: char = arg.chars().next().unwrap();
                if !none_okay || (none_okay && lead_char != '-') {
                    *i += 1;
                    Some(arg.as_str())
                } else if none_okay && lead_char == '-' {
                    None
                } else {
                    eprintln!("Expected argument after '{}' option", name);
                    help(ExitCode::ArgParseError);
                }
            } else if none_okay {
                None
            } else {
                *i += 1;
                Some("")
            }
        },
        None => if none_okay {
            None
        } else {
            eprintln!("Expected argument after '{}' option", name);
            help(ExitCode::ArgParseError);
        },
    }
}

fn parse_arg_complex<'a>(
    arg: &'a String,
    input: &mut InputType<'a>,
    output: &mut OutputType<'a>,
    options: &mut RunOptions,
) {
    if arg.len() > 1 && arg.chars().next().unwrap() == '-' {
        match arg.find('=') {
            None    => {
                eprintln!("Unrecognized argument '{}'", arg);
                help(ExitCode::ArgParseError);
            }
            Some(j) => {
                match &arg[0..j] {
                    "-e"        => *input = InputType::Expr(&arg[j + 1..]),
                    "--expr"    => *input = InputType::Expr(&arg[j + 1..]),
                    "-o"        => *output = OutputType::new(&arg[j + 1..]),
                    "-v"        => options.verbose = parse_verbose(Some(&arg[j + 1..])),
                    "--verbose" => options.verbose = parse_verbose(Some(&arg[j + 1..])),
                    _           => {
                        eprintln!("Unrecognized argument '{}'", arg);
                        help(ExitCode::ArgParseError);
                    }
                }
            }
        }
    } else if *input != InputType::None {
        eprintln!("Found more than one input ('{}' and '{}')", *input, arg);
        help(ExitCode::ArgParseError);
    } else if arg.len() == 1 && arg == "-" {
        *input = InputType::Stdin;
    } else {
        *input = InputType::File(arg.as_str());
    }
}

fn parse_verbose(mode: Option<&str>) -> Option<VerboseMode> {
    match mode {
        None        => Some(VerboseMode::All),
        Some(arg)   => Some(str_to_verbose_mode(arg)),
    }
}

fn print_pkg_info(should_exit: bool) {
    eprintln!("Welcome to {} version {}\n{}", PACKAGE, VERSION, LICENSE);
    if should_exit { exit(ExitCode::Ok); }
}

fn str_to_verbose_mode(s: &str) -> VerboseMode {
    match s.to_lowercase().as_str() {
        "all"       => VerboseMode::All,
        "lexer"     => VerboseMode::Lexer,
        "parser"    => VerboseMode::Parser,
        "sem"       => VerboseMode::Sem,
        "codegen"   => VerboseMode::CodeGen,
        _           => {
            eprintln!("Unexpected string for verbose mode: {}", s);
            exit(ExitCode::ArgParseError);
        }
    }
}

fn main() -> ! {
    let args: Vec<String> = env::args().collect();
    let mut name: String = "Stdin".to_string();
    let mut input: InputType = InputType::None;
    let mut output: OutputType = OutputType::Stdout;
    let mut options: RunOptions = RunOptions::new();

    parse_args(&args, &mut input, &mut output, &mut options);

    let mut tokens: Vec<Token> = Vec::new();
    match input {
        InputType::None     => {
            eprintln!("Unexpected input");
            exit(ExitCode::ArgParseError);
        }
        InputType::Stdin    => {
            let mut lex = Lexer::new(&name, stdin(), &options);
            Lexer::lex_input(&mut tokens, &mut lex, &options);
        }
        InputType::Expr(e)  => {
            let mut lex = Lexer::new(&name, Cursor::new(e.to_string()), &options);
            Lexer::lex_input(&mut tokens, &mut lex, &options);
        }
        InputType::File(f)  => {
            name = f.to_string();
            let file: File = File::open(f).expect("Failed to open input file");
            let mut lex = Lexer::new(&name, file, &options);
            Lexer::lex_input(&mut tokens, &mut lex, &options);
        }
    }

    let mut expr_tmp: Expr = Default::default();
    let mut ast: Box<&mut dyn Ast> = Box::new(&mut expr_tmp);
    let mut parser: Parser = Parser::new(&tokens, &options);
    Parser::parse_input(&mut ast, &mut parser, &name, &options);

    let sem_check: bool = Semantics::check_all(*ast, &options);
    assert!(sem_check);

    exit(ExitCode::Ok);
}
