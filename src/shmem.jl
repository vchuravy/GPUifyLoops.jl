__size(args::Tuple) = Tuple{args...}
__size(i::Int) = Tuple{i}

__shmem(D::Device, args...) = throw(MethodError(__shmem, (D, args...)))
@inline __shmem(::CPU, ::Type{T}, ::Val{dims}, ::Val) where {T, dims} =MArray{__size(dims), T}(undef)

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative

    @inline function __shmem(::CUDA, ::Type{T}, ::Val{dims}, ::Val{id}) where {T, dims, id}
        ptr = CUDAnative._shmem(Val(id), T, Val(prod(dims)))
        CUDAnative.CuDeviceArray(dims, CUDAnative.DevicePtr{T, CUDAnative.AS.Shared}(ptr))
    end
end

# TODO: AMDGPU shmem

shmem_id = 0
macro shmem(T, dims)
    global shmem_id
    id = shmem_id::Int += 1

    quote
        $__shmem($backend(), $(esc(T)), Val($(esc(dims))), Val($id))
    end
end
