# GPUifyLoops.jl

GPUifyLoops tries to solve the problem of code-duplication that can occur
when writing performant kernels that target multiple devices.

## API

```@docs
@loop
@setup
@synchronize
```

## Examples
### Simple

````@eval
using Markdown
Markdown.parse("""
```julia
$(read("../../examples/simple.jl", String))
```
""")
````

## Other useful tools
### Loop unrolling

On Julia `v1.2.0-DEV.462` we can pass information to the LLVM loop tooling.
GPUifyLoops contains a macro `@unroll` that can unroll a loop fully if the
trip count is known or partially by a factor.

```@docs
@unroll
```
#### Example:

```julia
@noinline iteration(i) = @show i
# Unknown loop count
f(N) = @unroll 3 for i in 1:N
    iteration(i)
end
@code_llvm f(10)
```

This should yield something like:
```LLVM
  %6 = call i64 @julia_iteration_12527(i64 %value_phi3)
  %7 = add nuw i64 %value_phi3, 1
  %8 = call i64 @julia_iteration_12527(i64 %7)
  %9 = add i64 %value_phi3, 2
  %10 = call i64 @julia_iteration_12527(i64 %9)
  %11 = add i64 %value_phi3, 3
```

You can also unroll a loop fully, but that requires a known/computable
trip-count:

```julia
@noinline iteration(i) = @show i
# Unknown loop count
f() = @unroll for i in 1:10
    iteration(i)
end
@code_llvm f()
```

Which yields something like:
```LLVM
  %4 = call i64 @julia_iteration_12527(i64 1)
  %5 = call i64 @julia_iteration_12527(i64 2)
  %6 = call i64 @julia_iteration_12527(i64 3)
  %7 = call i64 @julia_iteration_12527(i64 4)
  %8 = call i64 @julia_iteration_12527(i64 5)
  %9 = call i64 @julia_iteration_12527(i64 6)
  %10 = call i64 @julia_iteration_12527(i64 7)
  %11 = call i64 @julia_iteration_12527(i64 8)
  %12 = call i64 @julia_iteration_12527(i64 9)
  %13 = call i64 @julia_iteration_12527(i64 10)
```
