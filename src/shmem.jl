shmem_id = 0

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative

    @inline function _shmemcu(::Type{T}, ::Val{N}, ::Val{J}) where {T, N, J}
      len = prod(N)
      ptr = CUDAnative._shmem(J, T, len)
      CUDAnative.CuDeviceArray(N, CUDAnative.DevicePtr{T, CUDAnative.AS.Shared}(ptr))
    end
end

_shmem(::Type{T}, ::Val{N}, ::Val{J}) where {T, N, J} = nothing

Cassette.overdub(ctx::Ctx, ::typeof(_shmem), args...) = _shmemcu(args...)

macro shmem(T, Dims)
    global shmem_id
    id = shmem_id::Int += 1

    dims = Dims.args
    esc(quote
        if !$isdevice()
            $MArray{Tuple{$(dims...)}, $T}(undef)
        else
            _shmem(Val($T), Val(Dims), Val($id))
        end
    end)
end
