macro shmem(T, Dims)
    dims = Dims.args
    quote
        if $iscpu(__DEVICE)
            $MArray{Tuple{$(dims...)}, $T}(undef)
        else
            @cuStaticSharedMem($T, $Dims)
        end
    end
end
