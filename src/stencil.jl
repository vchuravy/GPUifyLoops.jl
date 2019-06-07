
struct Stencil{N, Dim, U}
    dims::Dim
    arrays::U
    Stencil{N}(dims, arrays) where N = new{N, typeof(dims), typeof(arrays)}(dims, arrays) 
end

abstract type StencilKind end
struct SevenPoint <: StencilKind end
struct Full <: StencilKind end

@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    include("custencil.jl")
end

# note this 3D only
function stencil(dims, kind, args::Vararg{<:Any, N}) where N
    if isdevice()
        return CuStencil.stencil(dims, kind, args...)
    else
        return Stencil{N}(dims, args)
    end
end

function Base.iterate(stencil::Stencil{N}) where N
    i, j, k = (1, 1, 1)
    I, J, K = stencil.dims
    inds = CartesianIndices((1:I, 1:J, 1:K))
    if !checkbounds(Bool, inds, i, j, k)
        return nothing
    end

    regions = ntuple(Val(N)) do ind
        data = stencil.arrays[ind]
        SArray{Tuple{3,3,3}}(view(data, i-1:i+1, j-1:j+1, k-1:k+1))
    end

    I = nextind(inds, CartesianIndex(i,j,k))
    ((i,j,k,regions...), I)
end

function Base.iterate(stencil::Stencil{N}, Idx) where N
    i, j, k = Idx.I
    I, J, K = stencil.dims
    inds = CartesianIndices((1:I, 1:J, 1:K))
    if !checkbounds(Bool, inds, i, j, k)
        return nothing
    end

    regions = ntuple(Val(N)) do ind
        data = stencil.arrays[ind]
        SArray{Tuple{3,3,3}}(view(data, i-1:i+1, j-1:j+1, k-1:k+1))
    end

    Idx = nextind(inds, Idx)
    ((i,j,k,regions...), Idx)
end
