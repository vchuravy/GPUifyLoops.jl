module CuStencil

import ..GPUifyLoops: SevenPoint, Full

using ..CUDAnative
using StaticArrays

# TODO:
# In theory we can reduce register usage, by shuffeling in the next iteration
# only tricky situation is the edges.

struct Stencil{N, Dim, Kind, Shmem, U}
    dims::Dim
    buf::Shmem
    arrays::U
    Stencil{N}(dims::Dim, kind::Kind, shmem::Shmem, arrays::U) where {N, Dim, Kind, Shmem, U} =
        new{N, Dim, Kind, Shmem, U}(dims, shmem, arrays) 
end

# note this 3D only
function stencil(dims, kind, args::Vararg{<:Any, N}) where N
    eltypes = map(eltype, args)
    T = eltypes[1]
    #T = reduce(promote_type, eltypes)
    @assert Base.isconcretetype(T)

    buf = @cuDynamicSharedMem T (blockDim().x+2, blockDim().y+2)
    Stencil{N}(dims, kind, buf, args)
end

function load_stencil!(::Type{Kind}, buf, data, i, j, k) where Kind
    # translate between local to global indices
    m = threadIdx().x
    n = threadIdx().y
    # halo region 
    m, n = m+1, n+1

    # boundary of block
    M = blockDim().x+1
    N = blockDim().y+1

    full = Kind <: Full

    @inbounds begin
        buf[m, n] =  data[i, j, k]

        if m == 2
            buf[m-1, n] = data[i-1, j, k]
            if full && n == 2
                buf[m-1, n-1] = data[i-1, j-1, k]
            elseif full && n == N 
                buf[m-1, n+1] = data[i-1, j+1, k]
            end
        elseif m == M
            buf[m+1, n] = data[i+1, j, k]
            if full && n == 2
                buf[m+1, n-1] = data[i+1, j-1, k]
            elseif full && n == N 
                buf[m+1, n+1] = data[i+1, j+1, k]
            end
        end

        if n == 2
            buf[m, n-1] = data[i, j-1, k]
        elseif n == N 
            buf[m, n+1] = data[i, j+1, k]
        end
    end

    sync_warp()
    return m, n
end


function load_slice(buf,m,n)
    # SMatrix{3,3}(view(buf, m-1:m+1, n-1:n+1))
    inds = CartesianIndices((m-1:m+1, n-1:n+1))
    data = ntuple(Val(9)) do i
        buf[inds[i]]
    end
    SMatrix{3,3}(data)
end

merge_slices(a,b,c) = SArray{Tuple{3,3,3}}(a.data..., b.data..., c.data...)

function Base.iterate(stencil::Stencil{N, Kind}) where {N, Kind}
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = 1
    I, J, K = stencil.dims
    if !(i in 1:I && j in 1:J && k in 1:K)
        return nothing
    end

    buf = stencil.buf

    regions = ntuple(Val(N)) do ind
        data = stencil.arrays[ind]
        # load the boundary
        m, n = load_stencil!(Kind, buf, data, i, j, k-1)
        pre = load_slice(buf, m, n)

        m, n = load_stencil!(Kind, buf, data, i, j, k)
        current = load_slice(buf, m, n)

        m, n = load_stencil!(Kind, buf, data, i, j, k+1)
        next = load_slice(buf, m, n)

        ldata = merge_slices(pre, current, next)
    end

    ((i,j,k,regions...), (regions, k+1))
end


function Base.iterate(stencil::Stencil{N, Kind}, (regions, k)) where {N, Kind}
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    I, J, K = stencil.dims
    if !(i in 1:I && j in 1:J && k in 1:K)
        return nothing
    end

    buf = stencil.buf

    next_regions = ntuple(Val(N)) do ind
        data = stencil.arrays[ind]
        old = regions[ind]

        m, n = load_stencil!(Kind, buf, data, i, j, k+1)
        next = load_slice(buf, m, n)

        ldata = merge_slices(old[:,:,2], old[:,:,3], next)
    end

    ((i,j,k, next_regions...), (next_regions, k+1))
end

end
