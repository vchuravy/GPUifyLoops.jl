module GPUifyLoops

using StaticArrays
using Requires

export @setup, @loop, @synchronize
export @scratch, @shmem

include("StructsOfArrays.jl")

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative
    @eval begin
        sync(::Val{:GPU}) = sync_threads()
        StructsOfArrays._type_with_eltype(::Type{CuDeviceArray{_T,_N,AS}}, T, N) where{_T,_N,AS} = CuDeviceArray(T,N,AS)
        StructsOfArrays._type(::Type{<:CuDeviceArray}) = CuDeviceArray
    end
end

@init @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
    using .CuArrays
    @eval begin
        StructsOfArrays._type_with_eltype(::Type{<:CuArray}, T, N) = CuArray{T, N}
        StructsOfArrays._type(::Type{<:CuArray}) = CuArray
    end
end


###
# Simple macros that help to write functions that run
# both on the CPU and GPU
###

iscpu(::Val{:GPU}) = false
iscpu(::Val{:CPU}) = true
sync(::Val{:CPU}) = nothing

macro setup(sym)
    esc(:(local __DEVICE = Val($sym)))
end

macro synchronize()
    esc(:($sync(__DEVICE)))
end

# TODO:
# - check if __DEVICE is defined

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

