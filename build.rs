// Copyright 2024, Giordano Salvador
// SPDX-License-Identifier: BSD-3-Clause

use std::env;
use std::fmt;
use std::path::PathBuf;

#[cfg_attr(target_os = "linux", path="src/command.rs")]
#[cfg_attr(target_os = "macos", path="src/command.rs")]
#[cfg_attr(target_os = "windows", path="src\\command.rs")]
mod command;
use command::Command;
use command::CommandResult;

#[cfg_attr(target_os = "linux", path="src/exit_code.rs")]
#[cfg_attr(target_os = "macos", path="src/exit_code.rs")]
#[cfg_attr(target_os = "windows", path="src\\exit_code.rs")]
mod exit_code;
use exit_code::exit;
use exit_code::ExitCode;

#[derive(Copy,Clone)]
enum GenType {
    Dialect,
    Ops,
    Opt,
    Types,
}

#[derive(Copy,Clone)]
enum ExtType {
    Header,
    HeaderInc,
    Impl,
    ImplInc,
    LibStatic,
    Object,
    TableGen,
}

#[derive(Copy,Clone)]
struct NameConfig(GenType, ExtType);

impl fmt::Display for GenType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            GenType::Dialect    => "dialect",
            GenType::Ops        => "ops",
            GenType::Opt        => "opt",
            GenType::Types      => "types",
        })
    }
}

impl fmt::Display for ExtType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", match self {
            ExtType::Header     => "h",
            ExtType::HeaderInc  => "h.inc",
            ExtType::Impl       => "cpp",
            ExtType::ImplInc    => "cpp.inc",
            ExtType::LibStatic  => "a",         // TODO: multi-platform?
            ExtType::Object     => "o",
            ExtType::TableGen   => "td",
        })
    }
}

impl NameConfig {
    pub fn new(gen: GenType, ext: ExtType) -> Self {
        NameConfig{0: gen, 1: ext}
    }
}

/// Processes tablegen files to generate C++ headers and modules for the given dialect
fn add_mlir_dialect(dialect: &str, dialect_namespace: &str, llvm_include_path: &PathBuf) -> () {
    let td_ops      = input(&get_name(dialect, NameConfig::new(GenType::Ops, ExtType::TableGen)));
    let h_ops       = output(&get_name(dialect, NameConfig::new(GenType::Ops, ExtType::HeaderInc)));
    let cpp_ops     = output(&get_name(dialect, NameConfig::new(GenType::Ops, ExtType::ImplInc)));
    let h_types     = output(&get_name(dialect, NameConfig::new(GenType::Types, ExtType::HeaderInc)));
    let cpp_types   = output(&get_name(dialect, NameConfig::new(GenType::Types, ExtType::ImplInc)));
    let h_dialect   = output(&get_name(dialect, NameConfig::new(GenType::Dialect, ExtType::HeaderInc)));
    let cpp_dialect = output(&get_name(dialect, NameConfig::new(GenType::Dialect, ExtType::ImplInc)));

    eprintln!("> Processing tablegen ops: {}", td_ops.display());
    let tblgen_ops_decls_result = Command::run("mlir-tblgen", &[
        "-I",
        llvm_include_path.to_str().unwrap(),
        "--gen-op-decls",
        "--write-if-changed",
        "-o",
        h_ops.to_str().unwrap(),
        td_ops.to_str().unwrap(),
    ], true);
    let _ = process_result(&tblgen_ops_decls_result);
    eprintln!();
    let tblgen_ops_defs_result = Command::run("mlir-tblgen", &[
        "-I",
        llvm_include_path.to_str().unwrap(),
        "--gen-op-defs",
        "--write-if-changed",
        "-o",
        cpp_ops.to_str().unwrap(),
        td_ops.to_str().unwrap(),
    ], true);
    let _ = process_result(&tblgen_ops_defs_result);
    eprintln!();
    let tblgen_types_decls_result = Command::run("mlir-tblgen", &[
        "-I",
        llvm_include_path.to_str().unwrap(),
        "--gen-typedef-decls",
        format!("--typedefs-dialect={}", dialect_namespace).as_str(),
        "--write-if-changed",
        "-o",
        h_types.to_str().unwrap(),
        td_ops.to_str().unwrap(),
    ], true);
    let _ = process_result(&tblgen_types_decls_result);
    eprintln!();
    let tblgen_types_defs_result = Command::run("mlir-tblgen", &[
        "-I",
        llvm_include_path.to_str().unwrap(),
        "--gen-typedef-defs",
        format!("--typedefs-dialect={}", dialect_namespace).as_str(),
        "--write-if-changed",
        "-o",
        cpp_types.to_str().unwrap(),
        td_ops.to_str().unwrap(),
    ], true);
    let _ = process_result(&tblgen_types_defs_result);
    eprintln!();
    let tblgen_dialect_decls_result = Command::run("mlir-tblgen", &[
        "-I",
        llvm_include_path.to_str().unwrap(),
        "--gen-dialect-decls",
        format!("--dialect={}", dialect_namespace).as_str(),
        "--write-if-changed",
        "-o",
        h_dialect.to_str().unwrap(),
        td_ops.to_str().unwrap(),
    ], true);
    let _ = process_result(&tblgen_dialect_decls_result);
    eprintln!();
    let tblgen_dialect_defs_result = Command::run("mlir-tblgen", &[
        "-I",
        llvm_include_path.to_str().unwrap(),
        "--gen-dialect-defs",
        format!("--dialect={}", dialect_namespace).as_str(),
        "--write-if-changed",
        "-o",
        cpp_dialect.to_str().unwrap(),
        td_ops.to_str().unwrap(),
    ], true);
    let _ = process_result(&tblgen_dialect_defs_result);
    eprintln!();
}

/// Returns a path to a static library generated for the dialect
/// Depends on generating the dialect via `add_mlir_dialect`
fn add_mlir_lib(dialect: &str, _dialect_namspace: &str, include_path: &PathBuf) -> PathBuf {
    let cpp_ops     = input(&get_name(dialect, NameConfig::new(GenType::Ops, ExtType::Impl)));
    let cpp_dialect = input(&get_name(dialect, NameConfig::new(GenType::Dialect, ExtType::Impl)));
    let obj_ops     = output(&get_name(dialect, NameConfig::new(GenType::Ops, ExtType::Object)));
    let obj_dialect = output(&get_name(dialect, NameConfig::new(GenType::Dialect, ExtType::Object)));
    let a_dialect   = output(&get_name(dialect, NameConfig::new(GenType::Dialect, ExtType::LibStatic)));

    let cxx_flags_result = Command::run("llvm-config", &["--cxxflags"], true);
    let cxx_flags = process_result(&cxx_flags_result).unwrap();

    let mut args_obj_ops = vec![
        "-I",
        include_path.to_str().unwrap(),
        "-c",
        "-o",
        obj_ops.to_str().unwrap(),
        cpp_ops.to_str().unwrap(),
    ];
    args_obj_ops.append(&mut cxx_flags.split(|c| {char::is_ascii_whitespace(&c)}).collect());
    let clang_obj_ops_result = Command::run("clang++", args_obj_ops.as_slice(), true);
    let _ = process_result(&clang_obj_ops_result);
    eprintln!();
    let mut args_obj_dialect = vec![
        "-I",
        include_path.to_str().unwrap(),
        "-c",
        "-o",
        obj_dialect.to_str().unwrap(),
        cpp_dialect.to_str().unwrap()
    ];
    args_obj_dialect.append(&mut cxx_flags.split(|c| {char::is_ascii_whitespace(&c)}).collect());
    let clang_obj_dialect_result = Command::run("clang++", args_obj_dialect.as_slice(), true);
    let _ = process_result(&clang_obj_dialect_result);
    eprintln!();
    let ar_dialect_result = Command::run("ar", &[
        "-r",
        a_dialect.to_str().unwrap(),
        obj_dialect.to_str().unwrap(),
        obj_ops.to_str().unwrap(),
    ], true);
    let _ = process_result(&ar_dialect_result);
    eprintln!();
    let ranlib_dialect_result = Command::run("ranlib", &[a_dialect.to_str().unwrap()], true);
    let _ = process_result(&ranlib_dialect_result);
    eprintln!();
    a_dialect
}

fn add_llvm_executable(
    exe_name: &str,
    cpp_source: &str,
    include_path: &PathBuf,
    libs: Vec<&str>,
) -> PathBuf {
    let cxx_flags_result = Command::run("llvm-config", &["--cxxflags"], true);
    let cxx_flags = process_result(&cxx_flags_result).unwrap();
    let ld_flags_result = Command::run("llvm-config", &["--ldflags"], true);
    let ld_flags = process_result(&ld_flags_result).unwrap();
    let libs_static_result = Command::run("llvm-config", &["--libfiles", "--link-static"], true);
    let libs_static = process_result(&libs_static_result).unwrap();
    let libs_shared_result = Command::run("llvm-config", &["--libfiles", "--link-shared"], true);
    let libs_shared = process_result(&libs_shared_result).unwrap();
    let mlir_libs = get_mlir_libs();
    let cpp_source_path = input(cpp_source);
    check_file(&cpp_source_path, cpp_source);
    let exe_path = output(exe_name);
    let mut args = vec![
        "-I",
        include_path.to_str().unwrap(),
        "-o",
        exe_path.to_str().unwrap(),
        cpp_source_path.to_str().unwrap(),
    ];
    args.append(&mut cxx_flags.split(|c| {char::is_ascii_whitespace(&c)}).collect());
    args.append(&mut ld_flags.split(|c| {char::is_ascii_whitespace(&c)}).collect());
    args.append(&mut libs_static.split(|c| {char::is_ascii_whitespace(&c)}).collect());
    args.append(&mut libs_shared.split(|c| {char::is_ascii_whitespace(&c)}).collect());
    args.append(&mut mlir_libs.iter().map(|p| {p.to_str().unwrap()}).collect());
    args.append(&mut libs.clone());
    let exe_result = Command::run("clang++", args.as_slice(), true);
    let _ = process_result(&exe_result);
    check_file(&exe_path, exe_name);
    exe_path
}

fn cargo_rerun(file: &str) -> () {
    println!("cargo::rerun-if-changed={}", file);
}

fn check_directory(path: &PathBuf, name: &str) -> () {
    if !path.exists() || !path.is_dir() {
        eprintln!("{} does not exist", name);
        exit(ExitCode::BuildError);
    } else {
        eprintln!();
    }
}

fn check_file(path: &PathBuf, name: &str) -> () {
    if !path.exists() || !path.is_file() {
        eprintln!("{} does not exist", name);
        exit(ExitCode::BuildError);
    } else {
        eprintln!();
    }
}

fn get_llvm_include_path() -> PathBuf {
    let llvm_includedir_result = Command::run("llvm-config", &["--includedir"], true);
    let llvm_include_str = process_result(&llvm_includedir_result);
    PathBuf::from(llvm_include_str.unwrap())
}

fn get_llvm_lib_path() -> PathBuf {
    let llvm_libdir_result = Command::run("llvm-config", &["--libdir"], true);
    let llvm_libdir_str = process_result(&llvm_libdir_result);
    PathBuf::from(llvm_libdir_str.unwrap())
}

fn get_mlir_libs() -> Vec<PathBuf> {
    let mut mlir_libs: Vec<PathBuf> = Vec::new();
    let mlir_lib_path = get_llvm_lib_path();
    let mlir_lib_prefix_path = mlir_lib_path.clone().join("libMLIR");
    let mlir_lib_prefix = mlir_lib_prefix_path.to_str().unwrap();
    for entry in mlir_lib_path.read_dir().expect("Failed to read mlir lib path") {
        match entry {
            Ok(entry)   => {
                let path = entry.path();
                if path.is_file() && path.to_str().unwrap().starts_with(mlir_lib_prefix) {
                    mlir_libs.push(path);
                }
            },
            _           => {
                eprintln!("Failed to read path in mlir lib path");
                exit(ExitCode::BuildError);
            },
        }
    }
    mlir_libs
}

fn get_name(dialect_namespace: &str, NameConfig(gen, ext): NameConfig) -> String {
    format!("{}-{}.{}", gen, dialect_namespace, ext)
}

fn input(file: &str) -> PathBuf {
    input_path().join(file)
}

fn input_path() -> PathBuf {
    let root_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(root_dir.as_str()).join("mlir")
}

fn output(file: &str) -> PathBuf {
    output_path().join(file)
}

fn output_path() -> PathBuf {
    let root_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let build_profile = env::var("PROFILE").unwrap();
    PathBuf::from(root_dir.as_str()).join("target").join(build_profile)
}

fn process_result(result: &CommandResult) -> Option<String> {
    if !result.success {
        exit(ExitCode::BuildError);
    }
    if result.stdout.is_some() {
        let out = result.stdout.as_ref().unwrap();
        eprintln!("{}", out);
        Some(out[..out.len()-1].to_string())
    } else {
        None
    }
}

#[allow(dead_code)]
fn build_dialect() -> () {
    cargo_rerun(&get_name("mlp", NameConfig::new(GenType::Dialect, ExtType::TableGen)));
    cargo_rerun(&get_name("mlp", NameConfig::new(GenType::Ops, ExtType::TableGen)));

    eprintln!("> Checking llvm-config installation");
    let llvm_config_result = Command::run("llvm-config", &["--version"], true);
    let _ = process_result(&llvm_config_result);

    eprintln!("> Getting llvm header installation");
    let llvm_include_path = get_llvm_include_path();
    let mlir_include_path = llvm_include_path.join("mlir");

    eprintln!("> Checking mlir header installation:\n{}", mlir_include_path.display());
    check_directory(&mlir_include_path, "mlir header installation");

    eprintln!("> Checking mlir-tblgen installation");
    let tblgen_version_result = Command::run("mlir-tblgen", &["--version"], true);
    let _ = process_result(&tblgen_version_result);

    eprintln!("> Generating mlp dialect");
    add_mlir_dialect("mlp", "mlp", &llvm_include_path);

    eprintln!("> Generating mlp library");
    let lib_mlp = add_mlir_lib("mlp", "mlp", &output_path());

    eprintln!("> Generating mlp-opt driver");
    let opt_source = get_name("mlp", NameConfig::new(GenType::Opt, ExtType::Impl));
    add_llvm_executable("mlp-opt", &opt_source, &output_path(), vec![lib_mlp.to_str().unwrap()]);
    exit(ExitCode::Ok);
}

fn main() -> () {
    exit(ExitCode::Ok);
}
