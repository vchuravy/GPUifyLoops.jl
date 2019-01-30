using GPUifyLoops
using Test
using Requires

kernel(A) = kernel(Val(:CPU), A)
function kernel(::Val{Dev}, A) where Dev
    @setup Dev

    @loop for i in (1:size(A,1);
                    threadIdx().x)
        A[i] = 2*A[i]
    end
    @synchronize
end

using Requires
@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative
    function kernel(A::CuArray)
        @cuda threads=128 kernel(Val(:GPU), A)
    end
end

if Base.find_package("CuArrays") !== nothing
    using CuArrays
    const DeviceArray = CuArray
else
    const DeviceArray = Array
end

data = DeviceArray{Float32}(undef, 1024)
kernel(data)

