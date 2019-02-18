module GPUifyLoops

using StaticArrays
using Requires

export @setup, @loop, @synchronize
export @scratch, @shmem

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative
end

###
# Simple macros that help to write functions that run
# both on the CPU and GPU
###

iscpu(::Val{:GPU}) = false
iscpu(::Val{:CPU}) = true
sync(::Val{:CPU}) = nothing
sync(::Val{:GPU}) = sync_threads()


"""
    @setup Dev

Setups some hidden state within the function that allows the other macros to
properly work.

```julia
function kernel(::Val{Dev}, A) where Dev
    @setup Dev
    # ...
end

kernel(A::Array) = kernel(Val(:CPU), A)
kernel(A::CuArray) = @cuda kernel(Val(:GPU), A)
```
"""
macro setup(sym)
    esc(:(local __DEVICE = Val($sym)))
end

"""
    @syncronize

Calls `sync_threads()` on the GPU and nothing on the CPU.
"""
macro synchronize()
    esc(:($sync(__DEVICE)))
end

# TODO:
# - check if __DEVICE is defined
"""
    @loop for i in (A; B)
        # body
    end

Take a `for i in (A; B)` expression and on the CPU lowers it to:

```julia
for i in A
    # body
end
```

and on the GPU:
```julia
for i in B
    if !(i in A)
        continue
    end
    # body
end
```
"""
macro loop(expr)
    if expr.head != :for
        error("Syntax error: @loop needs a for loop")
    end

    induction = expr.args[1]
    body = expr.args[2]

    if induction.head != :(=)
        error("Syntax error: @loop needs a induction variable")
    end

    rhs = induction.args[2]
    if rhs.head == :block
        @assert length(rhs.args) == 3
        # rhs[2] is a linenode
        cpuidx = rhs.args[1]
        gpuidx = rhs.args[3]

        rhs = Expr(:if, :($iscpu(__DEVICE)), cpuidx, gpuidx)
        induction.args[2] = rhs

        # use cpuidx calculation to check bounds of on GPU.
        bounds_chk = quote
            if !$iscpu(__DEVICE) && !($gpuidx in $cpuidx)
                continue
            end
        end 

        pushfirst!(body.args, bounds_chk)
    end

    return esc(Expr(:for, induction, body))
end

include("scratch.jl")
include("shmem.jl")

end

