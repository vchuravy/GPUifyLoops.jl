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

