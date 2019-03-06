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

f2(x) = sin(x)
f3(x) = 1 + f2(x)

kernel2!(A::Array, B::Array) = kernel2!(CPU(), A, B, f3)
function kernel2!(::Dev, A, B, h) where Dev
    @setup Dev
    @inbounds @loop for i in (1:size(A,1); threadIdx().x)
        A[i] = h(B[i])
    end
    nothing
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
        g(x) = GPUifyLoops.contextualize(CUDA(), f)(x)
        h(x) = GPUifyLoops.contextualize(CPU(), f)(x)
        @test g(3.0) == 6.0
        @test h(3.0) == 6.0
        f(x) = 3*x
        @test_broken g(3.0) == 9.0
        f1(x) = sin(x)
        g1(x) = GPUifyLoops.contextualize(CUDA(), f1)(x)
        asm = sprint(io->CUDAnative.code_llvm(io, g1, Tuple{Float64},
                                              dump_module=true))
        @test occursin("call double @__nv_sin", asm)

        begin
            data = rand(Float32, 1024)
            fdata = similar(data)
            kernel2!(fdata, data)

            @test f3.(data) ≈ fdata

            @eval function kernel2!(A::CuArray, B::CuArray)
                g3(x) = GPUifyLoops.contextualize(CUDA(), f3)(x)
                @cuda threads=length(A) kernel2!(CUDA(), A, B, g3)
            end

            cudata = CuArray(data)
            cufdata = similar(cudata)
            kernel2!(cufdata, cudata)

            @test f3.(data) ≈ cufdata
        end
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
