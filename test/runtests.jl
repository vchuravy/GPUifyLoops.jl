using GPUifyLoops
using Requires
using Test

kernel(A::Array) = kernel(Val(:CPU), A)
function kernel(::Val{Dev}, A) where Dev
    @setup Dev

    @loop for i in (1:size(A,1);
                    threadIdx().x)
        A[i] = 2*A[i]
    end
    @synchronize
end

@testset "Array" begin
    data = Array{Float32}(undef, 1024)
    kernel(data)
end

@static if Base.find_package("CuArrays") !== nothing
    using CuArrays
    using CUDAnative

    @eval function kernel(A::CuArray)
        @cuda threads=length(A) kernel(Val(:GPU), A)
    end

    @testset "CuArray" begin
        data = CuArray{Float32}(undef, 1024)
        kernel(data)
    end
end

