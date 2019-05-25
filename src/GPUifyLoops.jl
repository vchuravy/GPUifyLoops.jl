module GPUifyLoops

if VERSION < v"1.1"
    @error "GPUifyLoops depends on Julia v1.1"
end

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
export @unroll
export @launch

##
# contextualize
##
include("context.jl")

backend() = error("Calling backend function outside of context")
# FIXME: Get backend from Context or have Context per backend
Cassette.overdub(ctx::CuCtx, ::typeof(backend)) = CUDA()
Cassette.overdub(ctx::CPUCtx, ::typeof(backend)) = CPU()

macro launch(ex...)
    # destructure the `@launch` expression
    call = ex[end]
    kwargs = ex[2:end-1]

    device = ex[1]

    # destructure the kernel call
    if call.head != :call
        throw(ArgumentError("second argument to @launch should be a function call"))
    end

    f = call.args[1]
    args = call.args[2:end]

    quote
        $launch($(esc(device)), $(esc(f)), $(map(esc, args)...); $(map(esc, kwargs)...))
    end
end



"""
   launch(::Device, f, args..., kwargs...)

Launch a kernel on the GPU. `kwargs` are passed to `@cuda`
`kwargs` can be any of the compilation and runtime arguments
normally passed to `@cuda`.
"""
launch(dev::CPU, f, args...; kwargs...) = contextualize(dev, f)(args...)

"""
    launch_config(::F, maxthreads, args...; kwargs...)

Calculate a valid launch configuration based on the typeof(F), the
maximum number of threads, the functions arguments and the particular
launch configuration passed to the call.

Return a NamedTuple that has `blocks`, `threads`, `shmem`, and `stream`.
All arguments are optional, but blocks and threads is recommended.
"""
function launch_config(@nospecialize(f), maxthreads, args...; kwargs...)
    return kwargs
end

function split_kwargs(kwargs)
    compiler_kws = [:minthreads, :maxthreads, :blocks_per_sm, :maxregs]
    call_kws     = [:blocks, :threads, :shmem, :stream]
    compiler_kwargs = []
    call_kwargs = []
    for kwarg in kwargs
        key, val = kwarg
        if isa(key, Symbol)
            if key in compiler_kws
                push!(compiler_kwargs, kwarg)
            elseif key in call_kws
                push!(call_kwargs, kwarg)
            else
                throw(ArgumentError("unknown keyword argument '$key'"))
            end
        else
            throw(ArgumentError("non-symbolic keyword '$key'"))
        end
    end
    return compiler_kwargs, call_kwargs
end

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative

    function launch(dev::CUDA, f::F, args...; kwargs...) where F
        compiler_kwargs, call_kwargs = split_kwargs(kwargs)
        GC.@preserve args begin
            kernel_args = map(cudaconvert, args)
            kernel_tt = Tuple{map(Core.Typeof, kernel_args)...}
            kernel = cufunction(contextualize(dev, f), kernel_tt; compiler_kwargs...)

            maxthreads = CUDAnative.maxthreads(kernel)
            config = launch_config(f, maxthreads, args...; call_kwargs...)

            kernel(kernel_args...; config...)
        end
        return nothing
    end
end

isdevice(::CPU) = false
isdevice(::Device) = true
isdevice() = isdevice(backend())

sync(::CPU) = nothing
sync() = sync(backend())

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative
    sync(::CUDA) = CUDAnative.sync_threads()
end

@deprecate iscpu(::Val{:GPU}) isdevice()
@deprecate iscpu(::Val{:CPU}) !isdevice()
@deprecate sync(::Val{:GPU}) sync()
@deprecate sync(::Val{:CPU}) sync()


"""
    @syncronize

Calls `sync_threads()` on the GPU and nothing on the CPU.
"""
macro synchronize()
    :($sync())
end

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

        rhs = Expr(:if, :(!$isdevice()), cpuidx, gpuidx)
        induction.args[2] = rhs

        # use cpuidx calculation to check bounds of on GPU.
        bounds_chk = quote
            if $isdevice() && !($gpuidx in $cpuidx)
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

###
# Loopinfo
# - `@unroll`
###
include("loopinfo.jl")
using .LoopInfo

end
