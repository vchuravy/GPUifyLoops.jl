using GPUifyLoops
using Test

@testset "Unittests" begin
    include("test.jl")
end

# this test is broken within the testset
# test that redefinitions are propagated
# into contextualized functions.
f(x) = 2*x
g(x) = GPUifyLoops.contextualize(CUDA(), f)(x)
@test g(3.0) == 6.0
f(x) = 3*x
@test g(3.0) == 9.0

include("examples.jl")

