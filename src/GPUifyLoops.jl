module GPUifyLoops

abstract type Device end
struct CPU <: Device end

abstract type GPU <: Device end
struct CUDA <: GPU end

#=
# Hopefully we can eventually support AMDGPUs through ROCm
struct ROCm <: GPU end
=#

export CPU, CUDA, Device

using StaticArrays
using Requires

export @setup, @loop, @synchronize
export @scratch, @shmem
export contextualize

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative
end
include("context.jl")

iscpu(::GPU) = false
iscpu(::CPU) = true
sync(::CPU) = nothing
sync(::CUDA) = sync_threads()

@deprecate iscpu(::Val{:GPU}) iscpu(CUDA())
@deprecate iscpu(::Val{:CPU}) iscpu(CPU())
@deprecate sync(::Val{:GPU}) sync(CUDA())
@deprecate sync(::Val{:CPU}) sync(CPU())


"""
    @setup Dev

Setups some hidden state within the function that allows the other macros to
properly work.

```julia
function kernel(::Dev, A) where {Dev}
    @setup Dev
    # ...
end

kernel(A::Array) = kernel(CPU(), A)
kernel(A::CuArray) = @cuda kernel(GPU(), A)
```
"""
macro setup(sym)
    esc(:(local __DEVICE = $sym()))
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

###
# Scratch and shared-memory
###
include("scratch.jl")
include("shmem.jl")

end

