
_shmem(::Type{T}, ::Val{N}, ::Val{id}) where {T, N, id} = nothing
_shmemarray(A, shape) = nothing

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative

    @inline function Cassette.overdub(ctx::Ctx, ::typeof(_shmem), ::Type{T}, ::Val{N}, ::Val{id}) where {T, N, id}
        ptr = CUDAnative._shmem(Val(id), T, Val(N))
        CuDeviceArray(N, DevicePtr{T, CUDAnative.AS.Shared}(ptr))
    end
    @inline function Cassette.overdub(ctx::Ctx, ::typeof(_shmemarray), A::CUDAnative.CuDeviceArray, shape)
        CUDAnative.CuDeviceArray(shape, pointer(A))
    end
end

shmem_id = 0
macro shmem(T, Dims)
    global shmem_id
    id = shmem_id::Int += 1

    dims = Dims.args
    esc(quote
        if !$isdevice()
            $MArray{Tuple{$(dims...)}, $T}(undef)
        else
          ptr = $_shmem($T, Val($Dims), Val($id))
          $_shmemarray(ptr, $Dims)
        end
    end)
end
