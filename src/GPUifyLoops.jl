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

