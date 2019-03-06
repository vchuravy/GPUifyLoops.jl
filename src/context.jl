##
# Implements contextual dispatch through Cassette.jl
# Goals:
# - Rewrite common CPU functions to appropriate GPU intrinsics
#
# TODO:
# - error (erf, ...)
# - pow
# - min, max
# - mod, rem
# - gamma
# - bessel
# - distributions
# - unsorted

using Cassette

##
# Important for inference to not be able to see the constant value
##
@inline function unknowably_false()
    Base.llvmcall("ret i8 0", Bool, Tuple{})
end

##
# Forces inlining on everything that is not marked `@noinline`
# Cassette has a #265 issue, let's try to work around that.
##
function transform(ctx, ref)
    CI = ref.code_info
    noinline = any(@nospecialize(x) ->
                       Core.Compiler.isexpr(x, :meta) &&
                       x.args[1] == :noinline,
                   CI.code)
    CI.inlineable = true

    # 265 fix, insert a call to the original method
    # that we later will remove with LLVM's DCE
    self = GlobalRef(ref.method.module, ref.method.name)
    unknowably_false = GlobalRef(@__MODULE__, :unknowably_false)
    Cassette.insert_statements!(CI.code, CI.codelocs,
      (x, i) -> i == 1 ?  4 : nothing,
      (x, i) -> i == 1 ? [
          Expr(:call, Expr(:nooverdub, unknowably_false)),
          Expr(:gotoifnot, Core.SSAValue(i), i+3),
          Expr(:call, Expr(:nooverdub, self), [Core.SlotNumber(i) for i in 2:length(CI.slotnames)]...),
          x] : [x])
    return CI
end

const GPUifyPass = Cassette.@pass transform

Cassette.@context Ctx
const ctx = Cassette.disablehooks(Ctx(pass = GPUifyPass))

isdevice() = false
Cassette.overdub(ctx::Ctx, ::typeof(isdevice)) = true

###
# Cassette fixes
###
Cassette.overdub(::Ctx, ::typeof(Core.kwfunc), f) = return Core.kwfunc(f)

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative
    Cassette.overdub(::Ctx, ::typeof(CUDAnative.datatype_align), ::Type{T}) where {T} = CUDAnative.datatype_align(T)
end

###
# Rewrite functions
###

# libdevice.jl
const cudafuns = (:cos, :cospi, :sin, :sinpi, :tan,
          :acos, :asin, :atan,
          :cosh, :sinh, :tanh,
          :acosh, :asinh, :atanh,
          :log, :log10, :log1p, :log2,
          :exp, :exp2, :exp10, :expm1, :ldexp,
          :isfinite, :isinf, :isnan,
          :signbit, :abs,
          :sqrt, :cbrt,
          :ceil, :floor,)
for f in cudafuns
    @eval function Cassette.overdub(ctx::Ctx, ::typeof(Base.$f), x::Union{Float32, Float64})
        @Base._inline_meta
        return CUDAnative.$f(x)
    end
end

"""
    contextualize(::Dev, f)

This contexualizes the function `f` for a given device type `Dev`.

For the device `CUDA()`, `contextualize` replaces calls to math library
functions.  For example, `cos`, `sin`, are replaced with `CUDAnative.cos`,
`CUDAnative.sin`, respectively.

The full list functions that are replaced is $cudafuns.

# Examples
```julia
function kernel!(::Dev, A, f) where {Dev}
    @setup Dev
    @loop for i in (1:size(A,1); threadIdx().x)
        A[i] = f(A[i])
    end
end

g(x) = sin(x)
kernel!(A::Array) = kernel!(CPU(), A, contextualize(CPU(), g))
kernel!(A::CuArray) =
    @cuda threads=length(A) kernel!(CUDA(), A, contextualize(CUDA(), g))

a = rand(Float32, 1024)
b, c = copy(a), CuArray(a)

kernel!(b)
kernel!(c)

@assert g.(a) ≈ b
@assert g.(a) ≈ c
```
"""
function contextualize end
contextualize(::CUDA, f::F) where F = (args...) -> Cassette.overdub(ctx, f, args...)
contextualize(::CPU,  f::F) where F = (args...) -> f(args...)
