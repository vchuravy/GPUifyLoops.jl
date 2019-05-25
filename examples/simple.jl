using GPUifyLoops

function kernel(A)
    @loop for i in (1:size(A,1);
                    threadIdx().x)
        A[i] = 2*A[i]
    end
    @synchronize
end

data = Array{Float32}(undef, 1024)
@launch CPU() kernel(data)

@static if Base.find_package("CuArrays") !== nothing
    using CuArrays
    using CUDAnative

    kernel(A::CuArray) = @launch CUDA() kernel(A, threads=length(A))

    data = CuArray{Float32}(undef, 1024)
    kernel(data)
end

