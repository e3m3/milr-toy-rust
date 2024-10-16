#  Copyright

Copyright 2024, Giordano Salvador
SPDX-License-Identifier: BSD-3-Clause

Author/Maintainer:  Giordano Salvador <73959795+e3m3@users.noreply.github.com>


#  Description (MLIR Toy)
Learning rust by implementing the [MLIR Toy][1] [[1]] language using
    [MLIR (Multi-Level IR)][2] [[2]] compiler framework.
Baseline for the MLP (Multi-Level Program) language (using file extension ".mlp").

##  Language

The original license for MLIR Toy language can be found here [[3]].

### Lexer

```text
ident           ::= letter+ (letter | digit)*
number          ::= `.`? digit+ | (`0x` hex_digit+) | digit+ `.` digit*
digit           ::= [0-9]
hex_digit       ::= [a-fA-F0-9]
letter          ::= letter_lower | letter_upper | `_`
letter_lower    ::= [a-z]
letter_upper    ::= [A-Z]
whitespace      ::= ` ` | `\r` | `\n` | `\t`

any             ::= _
token           ::= { tokenkind, text }
tokenkind       ::=
    | Unknown
    | AngleL
    | AngleR
    | Assign
    | BraceL
    | BraceR
    | BracketL
    | BracketR
    | Comma
    | Comment
    | Colon
    | Def
    | DotStar
    | Eoi
    | Eol
    | Ident
    | Minus
    | Number
    | ParenL
    | ParenR
    | Plus
    | Print
    | Return
    | Semicolon
    | Slash
    | Star
    | Transpose
    | Var
text            ::=
    | ``
    | `,`
    | `/``/` any*
    | `:`
    | `;`
    | ident
    | `-`
    | number
    | `<`
    | `>`
    | `{`
    | `}`
    | `[`
    | `]`
    | `+`
    | `/`
    | `*`
    | `=`
    | `.*`
    | `def`
    | `print`
    | `return`
    | `transpose`
    | `var`
```

### Grammar

```text
return_stmt     ::= Return Semicolon | Return expr Semicolon
number_expr     ::= number
tensor_literal  ::= BracketL literal_list BracketR | number
literal_list    ::= tensor_literal | tensor_literal Comma literal_list
paren_expr      ::= ParenL expr ParenR
ident_expr      ::= ident | ident ParenL expr ParenR
primary         ::= ident_expr | number_expr | paren_expr | tensor_literal
binop           ::= Plus | Slash | DotStar | Star | Minus
binop_rhs       ::= ( binop primary )*
expr            ::= primary binop_rhs
type            ::= AngleL shape_list AngleR
shape_list      ::= number | number Comma shape_list
decl            ::= Var ident [ type ] Assign expr
block           ::= BraceL expr_list BraceR
expr_list       ::= block_expr Semicolon expr_list
block_expr      ::= decl | return_stmt | expr
prototype       ::= Def ident ParenL decl_list ParenR
decl_list       ::= ident | ident Comma decl_list
definition      ::= prototype block
```

##  Prerequisites

*   libstdc++

*   rust-2021

*   llvm-19 and llvm-sys (or llvm version matching llvm-sys)

*   llvm-19 and mlir-sys (or llvm version matching mlir-sys)

*   clang-19 (for executables and `-C|--c-main` flags)

*   python3-lit, FileCheck (for testing)

    *   By default, `tests/lit-tests.rs` will search for the lit executable in
        `$PYTHON_VENV_PATH/bin` (if it exists) or the system's `/usr/bin`.

*   [docker|podman] (for testing/containerization)

    *   A [Fedora][4] [[4]] image can be built using `containers/Containerfile.fedora*`.

##  Setup

*   Native build and test:
    
    ```shell
    cargo build
    cargo test -- --nocapture
    ```

*   Container build and test [podman][5] [[5]]:

    ```shell
    podman build -t calcc -f container/Containerfile .
    ```

*   Container build and test [docker][6] [[6]]:

    ```shell
    docker build -t calcc -f container/Dockerfile .
    ```

*   If `make` is installed, you can build the image by running:

    ```shell
    make
    ```

##   Usage

From the help message (`mlpc --help`):

```text
usage: mlpc [OPTIONS] <INPUT>
INPUT              '-' (i.e., Stdin) or a file path
OPTIONS:
--ast              Print the AST after parsing
-b|--bitcode       Output LLVM bitcode (post-optimization) (.bc if used with -o)
-B|--bytecode      Output MLIR bytecode (post-optimization) (.mlbc if used with -o)
-c                 Output an object file (post-optimization) (.o if used with -o)
--drop             Drop unknown tokens instead of failing
-e|--expr[=]<E>    Process expression E instead of INPUT file
-h|--help          Print this list of command line options
--lex              Exit after running the lexer
--ir               Exit after printing IR (pre-optimization)
--llvmir           Output LLVM IR (post-optimization) (.ll if used with -o)
-S|--mlir          Output MLIR IR (post-optimization) (.mlir if used with -o)
-k|--no-main       Omit linking with main module (i.e., output kernel only)
                   When this option is selected, an executable cannot be generated
--notarget         Omit target specific configuration in MLIR IR/bytecode
-o[=]<F>           Output to file F instead of Stdout ('-' for Stdout)
                   If no known extension is used (.bc|.exe|.ll|.mlbc|.mlir|.o) an executable is assumed
                   An executable requires llc and clang to be installed
-O<0|1|2|3>        Set the optimization level (default: O2)
--parse            Exit after running the parser
--sem              Exit after running the semantics check
-C|--c-main        Link with a C-derived main module (src/main.c.template)
                   This option is required for generating object files and executables on MacOS
                   and requires clang to be installed
-v|--verbose[[=]M] Enable verbose output (M=[all|lexer|parser|sem|codegen]; default: all)
--version          Display the package version and license information
```


#  References

[1]:    https://mlir.llvm.org/docs/Tutorials/Toy/

[2]:    https://mlir.llvm.org/

[3]:    https://github.com/llvm/llvm-project/blob/main/mlir/LICENSE.TXT

[4]:    https://fedoraproject.org/

[5]:    https://podman.io/

[6]:    https://www.docker.com/

1.  `https://mlir.llvm.org/docs/Tutorials/Toy/`

1.  `https://mlir.llvm.org/`

1.  `https://github.com/llvm/llvm-project/blob/main/mlir/LICENSE.TXT`

1.  `https://fedoraproject.org/`

1.  `https://podman.io/`

1.  `https://www.docker.com/`
