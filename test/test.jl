using GPUifyLoops
using Test
using InteractiveUtils

function kernel(A)
    @loop for i in (1:size(A,1);
                    threadIdx().x)
        A[i] = 2*A[i]
    end
    @synchronize
end

f2(x) = sin(x)
f3(x) = 1 + f2(x)
f4(x) = x^1.2
f5(x) = x^3

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

        @test g(3.0) == 9.0
        f1(x) = (sin(1.0 + x); return nothing)
        g1(x) = GPUifyLoops.contextualize(f1)(x)
        asm = sprint(io->CUDAnative.code_llvm(io, g1, Tuple{Float64}, kernel=true,
                                              optimize=false, dump_module=true))
        @test occursin(r"call .* double @__nv_sin", asm)
        @test occursin("fadd contract double", asm)

        @testset "don't overdub intrinsics" begin
            global simple_kernel, kernel
            simple_kernel(A, x) = (A[1] = 1 + x; return nothing)
            kernel(A, x) = GPUifyLoops.contextualize(simple_kernel)(A, x)
            CI, ret = CUDAnative.code_typed(kernel, Tuple{CUDAnative.CuDeviceArray{Int64,1, CUDAnative.AS.Global}, Int64}, debuginfo=:source)[1]

            intrinsics = findall(CI.code) do stmt
                if Base.Meta.isexpr(stmt, :call)
                    f = stmt.args[1]
                    if f isa GlobalRef
                        f = getfield(f.mod, f.name)
                        return f isa Core.IntrinsicFunction || f isa Core.Builtin
                    end
                end
                return false
            end

            for i in intrinsics
                lineinfo = CI.linetable[CI.codelocs[i]]
                @test !(lineinfo.method === :call ||
                        lineinfo.file === Symbol("context.jl"))
            end
        end

        begin
            global kernel2!
            data = rand(Float32, 1024)
            fdata = similar(data)

            kernel2!(fdata, data, f3)
            @test f3.(data) ≈ fdata

            kernel2!(fdata, data, f4)
            @test f4.(data) ≈ fdata

            kernel2!(fdata, data, f5)
            @test f5.(data) ≈ fdata

            function kernel2!(A::CuArray, B::CuArray, f)
                @launch CUDA() threads=length(A) kernel2!(A, B, f)
            end

            cudata = CuArray(data)
            cufdata = similar(cudata)

            kernel2!(cufdata, cudata, f3)
            @test f3.(data) ≈ cufdata

            kernel2!(cufdata, cudata, f4)
            @test f4.(data) ≈ cufdata

            kernel2!(cufdata, cudata, f5)
            @test f5.(data) ≈ cufdata
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

let
    function ker1!(::Val{Nq}) where Nq
      s_x = @shmem Float32 Nq
    end
    
    CI, rt = @code_typed ker1!(Val(10))
    @test Base.isconcretetype(rt)
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

function kernel_scratch(A, ::Val{N}) where N
    a = @scratch eltype(A) (N, N) 2
    b = @scratch eltype(A) (2, N, N) 2
    @loop for j in (1:size(A,2); threadIdx().y)
        @loop for i in (1:size(A,1); threadIdx().x)
            a[i, j] = A[i, j]
            b[1, i, j] = -A[i, j]
            b[2, i, j] = 2a[i, j]
        end
    end
end


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
    N = 10
    A = rand(N, N)
    @launch CPU() kernel_scratch(A, Val(N))

    @static if Base.find_package("CuArrays") !== nothing
        using CuArrays

        d_A = CuArray(A)
        @launch CUDA() kernel_scratch(d_A, Val(N))
    end
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

using StaticArrays
function kernel_MArray!(A)
  l_F = MArray{Tuple{3, 3}, eltype(A)}(undef)
  @inbounds for j = 1:3, i = 1:3
    l_F[i, j] = A[i, j]
  end
  nothing
end
function kernel_similar_MArray!(A)
  l_F = MArray{Tuple{3, 3}, eltype(A)}(undef)
  l_G = similar(l_F, Size(2,2))
  @inbounds for j = 1:2, i = 1:2
    l_G[i, j] = A[i, j]
  end
  nothing
end

@testset "StaticArrays" begin
  @static if Base.find_package("CuArrays") !== nothing
    using CuArrays
    using CUDAnative

    A = CuArray(rand(3,3))
    @launch CUDA() threads=(3,3) kernel_MArray!(A)
    @launch CUDA() threads=(3,3) kernel_similar_MArray!(A)
  end

  A = rand(3,3)
  @launch CPU() threads=(3,3) kernel_MArray!(A)
  @launch CPU() threads=(3,3) kernel_similar_MArray!(A)
end

# For the next three test we check that the _apply got inlined correctly
Base.@pure pure_f1(x, y) = x + y
let
    CI, rt = @code_typed GPUifyLoops.Cassette.overdub(GPUifyLoops.ctx, pure_f1, 1, 2)
    expr = CI.code[end-1]
    @test expr.head === :call || expr.head === :invoke
    @test expr.args[1] === GlobalRef(Base, :add_int)
end

Base.@pure pure_f2(x, ys...) = x + sum(ys)
let
    CI, rt = @code_typed GPUifyLoops.Cassette.overdub(GPUifyLoops.ctx, pure_f2, 1, 2, 3, 4)
    expr = CI.code[end-1]
    @test expr.head === :call || expr.head === :invoke
    @test expr.args[1] === GlobalRef(Base, :add_int)
end

Base.@pure pure_f3(ys...) = sum(ys)
let
    CI, rt = @code_typed GPUifyLoops.Cassette.overdub(GPUifyLoops.ctx, pure_f3, 1, 2, 3, 4)
    expr = CI.code[end-1]
    @test expr.head === :call || expr.head === :invoke
    @test expr.args[1] === GlobalRef(Base, :add_int)
end

