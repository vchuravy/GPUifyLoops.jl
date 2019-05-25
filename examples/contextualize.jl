using GPUifyLoops

f1(x) = sin(x)
f(x) = 1 + f1(x)

function kernel!(A, B, h)
    @inbounds @loop for i in (1:size(A,1); threadIdx().x)
        A[i] = h(B[i])
    end
    nothing
end

data = rand(Float32, 1024)
fdata = similar(data)
@launch CPU() kernel!(fdata, data, f)

@assert f.(data) ≈ fdata

@static if Base.find_package("CuArrays") !== nothing
    using CuArrays
    using CUDAnative

    function kernel!(A::CuArray, B::CuArray)
        @launch CUDA() threads=length(A) kernel!(A, B, f)
    end

    cudata = CuArray(data)
    cufdata = similar(cudata)
    kernel!(cufdata, cudata)

    @assert f.(data) ≈ cufdata
end
