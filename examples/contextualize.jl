using GPUifyLoops

f1(x) = sin(x)
f(x) = 1 + f1(x)

kernel!(A::Array, B::Array) = kernel!(CPU(), A, B, f)
function kernel!(::Dev, A, B, h) where Dev
    @setup Dev
    @inbounds @loop for i in (1:size(A,1); threadIdx().x)
        A[i] = h(B[i])
    end
    nothing
end

data = rand(Float32, 1024)
fdata = similar(data)
kernel!(fdata, data)

@assert f.(data) ≈ fdata

@static if Base.find_package("CuArrays") !== nothing
    using CuArrays
    using CUDAnative

    @eval function kernel!(A::CuArray, B::CuArray)
        g(x) = GPUifyLoops.contextualize(CUDA(),f)(x)
        @cuda threads=length(A) kernel!(CUDA(), A, B, g)
    end

    cudata = CuArray(data)
    cufdata = similar(cudata)
    kernel!(cufdata, cudata)

    @assert f.(data) ≈ cufdata
end
