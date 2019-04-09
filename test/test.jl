using GPUifyLoops
using Test

function kernel(A)
    @loop for i in (1:size(A,1);
                    threadIdx().x)
        A[i] = 2*A[i]
    end
    @synchronize
end

f2(x) = sin(x)
f3(x) = 1 + f2(x)

function kernel2!(A, B, h)
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

    function kernel(A::CuArray)
        @launch CUDA() threads=length(A) kernel(A)
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

        if GPUifyLoops.INTERACTIVE
            @test g(3.0) == 9.0
        else
            @test_broken g(3.0) == 9.0
        end
        f1(x) = (sin(x); return nothing)
        g1(x) = GPUifyLoops.contextualize(f1)(x)
        asm = sprint(io->CUDAnative.code_llvm(io, g1, Tuple{Float64},
                                              dump_module=true))
        @test occursin("call double @__nv_sin", asm)

        begin
            global kernel2!
            data = rand(Float32, 1024)
            fdata = similar(data)
            kernel2!(fdata, data, f3)

            @test f3.(data) ≈ fdata

            function kernel2!(A::CuArray, B::CuArray, f)
                @launch CUDA() threads=length(A) kernel2!(A, B, f)
            end

            cudata = CuArray(data)
            cufdata = similar(cudata)
            kernel2!(cufdata, cudata, f3)

            @test f3.(data) ≈ cufdata
        end
    end
end

function kernel3!(A)
    s1 = @shmem eltype(A) (1024,)
    s2 = @shmem eltype(A) (1024,)

    @loop for i in (1:size(A,1); threadIdx().x)
        s1[i] = 2*A[i]
        s2[i] = 3*A[i]
    end
    @synchronize
    @loop for i in (1:size(A,1); threadIdx().x)
        A[i] = s1[i]
    end
    nothing
end

@testset "shared memory" begin
    data = rand(Float32, 1024)
    cpudata = copy(data)

    @launch CPU() kernel3!(cpudata)
    @test cpudata ≈ 2 .* data

    @static if Base.find_package("CuArrays") !== nothing
        using CuArrays
        using CUDAnative

        cudata = CuArray(data)
        @launch CUDA() threads=length(cudata) kernel3!(cudata)
        @test Array(cudata) ≈ 2 .* data
    end
end

# Scratch arrays

function f1()
    A = @scratch Int64 (12, 3) 2
    @test A.data isa GPUifyLoops.MArray
    @test size(A.data) == (1,)
end

function f2()
    A = @scratch Int64 (12, 3) 1
    @test A.data isa GPUifyLoops.MArray
end

function f3()
    A = @scratch Int64 (12, 3) 1
    @test A isa GPUifyLoops.MArray
end

@testset "Scratch Arrays" begin
    contextualize(f1)()
    contextualize(f2)()
    f3()
end

@testset "Loopinfo" begin
    # Right now test that we don't break things
    # Should probably test that codegen is correct.
    f(N) = @unroll 2 for i in 1:N
        @show i
    end
    f(10)

    f() = @unroll for i in 1:10
        @show i
    end
    f()
end
