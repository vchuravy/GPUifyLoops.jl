using GPUifyLoops
using Test

kernel(A::Array) = kernel(CPU(), A)
function kernel(::Dev, A) where Dev
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
        @cuda threads=length(A) kernel(CUDA(), A)
    end

    @testset "CuArray" begin
        data = CuArray{Float32}(undef, 1024)
        kernel(data)
    end

    @testset "contextualize" begin
        f(x) = 2*x
        g(x) = GPUifyLoops.contextualize(f)(x)
        @test g(3.0) == 6.0
        f(x) = 3*x
        @test_broken g(3.0) == 9.0
        f1(x) = sin(x)
        g(x) = GPUifyLoops.contextualize(f1)(x)
        asm = sprint(io->CUDAnative.code_ptx(io, g, Tuple{Float64}))
        # TODO check the device function is called
    end
end

# Scratch arrays

function f1()
    @setup CUDA
    A = @scratch Int64 (12, 3) 2
    @test A.data isa GPUifyLoops.MArray
    @test size(A.data) == (1,)
end

function f2()
    @setup CUDA
    A = @scratch Int64 (12, 3) 1
    @test A.data isa GPUifyLoops.MArray
end

function f3()
    @setup CPU
    A = @scratch Int64 (12, 3) 1
    @test A isa GPUifyLoops.MArray
end

@testset "Scratch Arrays" begin
    f1()
    f2()
    f3()
end
