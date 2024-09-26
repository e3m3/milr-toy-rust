#  Copyright

Copyright 2024, Giordano Salvador
SPDX-License-Identifier: BSD-3-Clause

Author/Maintainer:  Giordano Salvador <73959795+e3m3@users.noreply.github.com>


#  Description (MLIR Toy)
Learning rust by implementing the [MLIR Toy][1] [[1]] language using
    [MLIR (Multi-Level IR)][2] [[2]] compiler framework.
Baseline for the MLP (Multi-Level Program) language (using file extension ".mlp").

##  Language

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
    | BraceL
    | BraceR
    | BracketL
    | BracketR
    | Comma
    | Comment
    | Colon
    | Def
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
    | `def`
    | `print`
    | `return`
    | `transpose`
    | `var`
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

    *   A [Fedora][3] [[3]] image can be built using `containers/Containerfile.fedora*`.

##  Setup

*   Native build and test:
    
    ```shell
    cargo build
    cargo test -- --nocapture
    ```

*   Container build and test [podman][4] [[4]]:

    ```shell
    podman build -t calcc -f container/Containerfile .
    ```

*   Container build and test [docker][5] [[5]]:

    ```shell
    docker build -t calcc -f container/Dockerfile .
    ```

*   If `make` is installed, you can build the image by running:

    ```shell
    make
    ```


#  References

[1]:    https://mlir.llvm.org/docs/Tutorials/Toy/

[2]:    https://mlir.llvm.org/

[3]:    https://fedoraproject.org/

[4]:    https://podman.io/

[5]:    https://www.docker.com/

1.  `https://mlir.llvm.org/docs/Tutorials/Toy/`

1.  `https://mlir.llvm.org/`

1.  `https://fedoraproject.org/`

1.  `https://podman.io/`

1.  `https://www.docker.com/`
