# mohu-dtype

Data types and type promotion system for the `mohu` scientific computing library.

`mohu-dtype` is a core foundational crate in the `mohu` workspace. Its main job is to manage the different data types (like integers, floats, and complex numbers) that our N-dimensional arrays can hold. 

It handles:
1. **Representing types** at runtime and compile time.
2. **Type promotion rules** (deciding what type you get when you add or multiply different types, e.g., `int32` + `float32`).
3. **Casting values** from one type to another safely.
4. **Ecosystem compatibility** (helping `mohu` share data with Python, NumPy, DLPack, and Apache Arrow without copying the actual data).

---

## 1. supported data types (`DType`)

In `mohu`, all supported data types are defined in a single, simple enum called `DType`. We support **15 primary types** covering booleans, integers, half-precision floats, standard floats, and complex numbers:

| Category | DType | Rust Primitive Type | NumPy Equivalent |
|---|---|---|---|
| **Boolean** | `Bool` | `bool` | `bool` |
| **Signed Integers** | `I8`, `I16`, `I32`, `I64` | `i8`, `i16`, `i32`, `i64` | `int8`, `int16`, `int32`, `int64` |
| **Unsigned Integers** | `U8`, `U16`, `U32`, `U64` | `u8`, `u16`, `u32`, `u64` | `uint8`, `uint16`, `uint32`, `uint64` |
| **Floats** | `F16`, `BF16` (Brain Float), `F32`, `F64` | `half::f16`, `half::bf16`, `f32`, `f64` | `float16`, `bfloat16`, `float32`, `float64` |
| **Complex Numbers** | `C64`, `C128` | `Complex<f32>`, `Complex<f64>` | `complex64`, `complex128` |

### compile-time trait hierarchy

At compile time, every supported primitive type implements a set of capabilities through a sealed hierarchy of traits. This guarantees memory layout safety and enables static optimization for compute operations:

```text
Scalar (Base trait: copyable, has size, zero/one representations)
  ├── RealScalar (Ordered: support for comparisons, min/max, absolute value)
  │     ├── IntScalar (Integers: bitwise operations, overflowing checks)
  │     │     ├── SignedScalar (Signed integers: negation, sign check)
  │     │     └── UnsignedScalar (Unsigned integers)
  │     └── FloatScalar (Floats: trigonometric, exponentials, NaN/Inf checks)
  └── ComplexScalar (Complex numbers: conjugate, norm/magnitude)
```

---

## 2. type promotion

When you perform an operation on two arrays with different types, `mohu` needs to decide what type the output array should be. 

This crate implements **NumPy-compatible type promotion rules**:
* **Widest type wins**: Combining `i16` (2 bytes) and `i32` (4 bytes) results in `i32`.
* **Float beats integer**: Combining `i32` and `f32` promotes to a float (`f64`) to preserve precision.
* **Complex beats real**: Combining any number with a complex number results in a complex number.

### standard promotion example

At runtime, these lookups are incredibly fast because they are pre-calculated in a 15×15 grid at compile time:

```rust
use mohu_dtype::{DType, promote};

// i32 + f32 results in f64
let result = promote(DType::I32, DType::F32);
assert_eq!(result, DType::F64);

// f16 + f32 results in f32
let result = promote(DType::F16, DType::F32);
assert_eq!(result, DType::F32);
```

---

## 3. type casting

Type casting is the process of converting a value from one data type to another. `mohu-dtype` supports three different modes of casting:

| Casting Mode | Description | Allowed Conversions (Examples) | Blocked Conversions (Examples) |
|---|---|---|---|
| **`Safe`** | Guaranteed to never lose any precision or information. | `i8` -> `i16`, `f32` -> `f64` | `f64` -> `f32` (narrowing), `f32` -> `i32` |
| **`SameKind`** | Allows conversions within the same group, even with precision loss. | `f64` -> `f32`, `i64` -> `i32` | `f32` -> `i32` (cross-kind conversion) |
| **`Unsafe`** | Allows any conversion (potentially truncating or lossy). | `f32` -> `i32`, `Complex` -> `Real` | None (all conversions are permitted) |

```rust
use mohu_dtype::{cast::cast_scalar, promote::CastMode};

// A Safe cast is guaranteed to work:
let value: f64 = cast_scalar::<i16, f64>(42_i16, CastMode::Safe).unwrap();
assert_eq!(value, 42.0);

// An Unsafe cast will truncate decimals:
let truncated: i32 = cast_scalar::<f64, i32>(5.99, CastMode::Unsafe).unwrap();
assert_eq!(truncated, 5);
```

---

## 4. external integrations

`mohu-dtype` provides native helper functions to translate our internal types into representations used by adjacent ecosystem libraries. This enables **zero-copy interop** (sharing memory directly without copying data back and forth):

| Ecosystem System | Compatibility Standard | Notes |
|---|---|---|
| **Python `struct`** | Single-character format codes | Codes like `'? '`, `'b'`, `'f'`, `'d'` for Python FFI. |
| **Python Buffer Protocol** | PEP 3118 format strings | Endian-prefixed formats (e.g., `"<f"`, `"<Zd"`) for direct memory mapping. |
| **DLPack Standard** | `DLDataType` structure | Standard format for tensor sharing with PyTorch, JAX, and NumPy. |
| **Apache Arrow** | `DataType` enum | Zero-copy hands-offs for analytical tools like `Polars` (`feature = "arrow"`). |

---

## 5. contributing & testing

If you are developing or contributing to `mohu-dtype`, you can run the following standard Cargo commands to verify your changes, run tests, and format your code:

```sh
# Type check the crate and workspace
cargo check --workspace --all-features

# Run the unit tests specifically for this crate
cargo test -p mohu-dtype --all-features

# Run Clippy (the Rust linter) to check for code quality issues
cargo clippy -p mohu-dtype --all-targets -- -D warnings

# Automatically format the code to match standard style guidelines
cargo fmt --all
```
