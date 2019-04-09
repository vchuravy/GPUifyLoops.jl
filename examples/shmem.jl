using GPUifyLoops

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

data = rand(Float32, 1024)
cpudata = copy(data)

@launch CPU() kernel3!(cpudata)
@assert cpudata ≈ 2 .* data

@static if Base.find_package("CuArrays") !== nothing
  using CuArrays
  using CUDAnative

  cudata = CuArray(data)
  @launch CUDA() threads=length(cudata) kernel3!(cudata)
  @assert Array(cudata) ≈ 2 .* data
end
