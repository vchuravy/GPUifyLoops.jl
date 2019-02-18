"""
Creates a local static memory shared inside one block.
Equivalent to `__local` of OpenCL or `__shared__ (<variable>)` of CUDA.
"""
macro LocalMemory(state, T, N)
    id = (shmem_counter[] += 1)
    quote
        lémem = LocalMemory($(esc(state)), $(esc(T)), Val($(esc(N))), Val($id))
        AbstractDeviceArray(lémem, $(esc(N)))
    end
end

export @LocalMemory

"""
Creates a block local array pointer with `T` being the element type
and `N` the length. Both T and N need to be static! C is a counter for
approriately get the correct Local mem id in CUDAnative.
This is an internal method which needs to be overloaded by the GPU Array backends
"""
function LocalMemory(state, ::Type{T}, ::Val{N}, ::Val{C}) where {N, T, C}
    error("Not implemented")
end

macro shmem(T, Dims)
    dims = Dims.args
    esc(quote
        if $iscpu(__DEVICE)
            $MArray{Tuple{$(dims...)}, $T}(undef)
        else
            @cuStaticSharedMem($T, $Dims)
        end
    end)
end
