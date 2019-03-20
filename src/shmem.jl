macro shmem(T, Dims)
    dims = Dims.args
    esc(quote
        if !$isdevice()
            $MArray{Tuple{$(dims...)}, $T}(undef)
        else
            @cuStaticSharedMem($T, $Dims)
        end
    end)
end
