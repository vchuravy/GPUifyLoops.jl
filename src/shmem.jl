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
            ptr = $GPUifyLoops.CUDAnative._shmem(Val($id), $T, Val(len))
            ptr = $GPUifyLoops.CUDAnative.DevicePtr{$T, $GPUifyLoops.CUDAnative.AS.Shared}(ptr)
            $GPUifyLoops.CUDAnative.CuDeviceArray($Dims, ptr)
        end
    end)
end
