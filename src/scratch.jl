using StaticArrays

"""
   @scratch T Dims M

Allocates scratch memory.
- `T` type of array
- `Dims` is a tuple of array dimensions
- `M` the number of dimensions at the tail that are implicit on the GPU
"""
macro scratch(T, Dims, M)
    @assert Dims.head == :tuple
    dims = Dims.args
    N = length(dims) - M
    gpudims = ntuple(i->dims[i], N)
    esc(quote 
        if $iscpu(__DEVICE)
            $MArray{Tuple{$(dims...)}, $T}(undef)
        else
            data = if $(length(gpudims)) > 0
                $ScratchArray{$N}(
                    $MArray{Tuple{$(gpudims...)}, $T}(undef)
                )
            else
                $ScratchArray{$N,$T}()
            end
        end
    end)
end

struct ScratchArray{N, D}
    data::D
    ScratchArray{N}(data::D) where {N, D} = new{N, D}(data)
    ScratchArray{N, T}() where {N, T} = new{N, T}(data)
end


Base.@propagate_inbounds function Base.getindex(A::ScratchArray{N}, I...) where N
    nI = ntuple(i->I[i], N)
    if nI == ()
        return A.data
    end
    return A.data[nI...]
end

Base.@propagate_inbounds function Base.setindex!(A::ScratchArray{N}, val, I...) where N
    nI = ntuple(i->I[i], N)
    if nI == ()
        return A.data = val
    end
    A.data[nI...] = N
end

