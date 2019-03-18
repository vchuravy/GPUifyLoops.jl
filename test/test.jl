using GPUifyLoops
using Test
using InteractiveUtils

kernel(A::Array) = kernel(Val(:CPU), A)
function kernel(::Val{Dev}, A) where Dev
    @setup Dev

    @loop for i in (1:size(A,1);
                    threadIdx().x)
        A[i] = 2*A[i]
    end
    @synchronize
end

function testprefetch(data, index)
    @test_nowarn prefetch(data, index)
    io = IOBuffer()
    code_llvm(io, prefetch, Tuple{typeof(data), typeof(index)})
    str = String(take!(io))
    @test occursin(r"call void @llvm\.prefetch\(i8\* %\d+, i32 0, i32 3, i32 1\)", str)
end

@testset "Array" begin
    data = Array{Float32}(undef, 1024)
    kernel(data)
    testprefetch(data, 12)
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
        testprefetch(data, 12)
    end
end

# Scratch arrays

function f1()
    @setup :GPU
    A = @scratch Int64 (12, 3) 2
    @test A.data isa GPUifyLoops.MArray
    @test size(A.data) == (1,)
end

function f2()
    @setup :GPU
    A = @scratch Int64 (12, 3) 1
    @test A.data isa GPUifyLoops.MArray
end

function f3()
    @setup :CPU
    A = @scratch Int64 (12, 3) 1
    @test A isa GPUifyLoops.MArray
end

@testset "Scratch Arrays" begin
    f1()
    f2()
    f3()
end
