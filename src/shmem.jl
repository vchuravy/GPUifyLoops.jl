shmem_id = 0

macro shmem(T, Dims)
    global shmem_id
    id = shmem_id::Int += 1

    dims = Dims.args
    esc(quote
        if !$isdevice()
            $MArray{Tuple{$(dims...)}, $T}(undef)
        else
            len = prod($Dims)
            ptr = CUDAnative._shmem(Val($id), $T, Val(len))
            CUDAnative.CuDeviceArray($Dims, CUDAnative.DevicePtr{$T, CUDAnative.AS.Shared}(ptr))
        end
    end)
end
