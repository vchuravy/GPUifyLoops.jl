using StaticArrays
using CUDAnative
using GPUifyLoops
using OffsetArrays
using CuArrays
using Adapt

# TODO: Add an option to load in the full 3x3x3x stencil including
# the 1,1 and I+1,I+1 corners
Adapt.adapt_structure(to, x::OffsetArray) = OffsetArray(Adapt.adapt(to, parent(x)), x.offsets)

struct Stencil{N, Dim, Shmem, U}
    dims::Dim
    buf::Shmem
    arrays::U
    Stencil{N}(dims, shmem, arrays) where N = new{N, typeof(dims), typeof(shmem), typeof(arrays)}(dims, shmem, arrays) 
end

# note this 3D only
function stencil(dims, args::Vararg{<:Any, N}) where N
    eltypes = map(eltype, args)
    T = reduce(promote_type, eltypes)
    @assert Base.isconcretetype(T)

    buf = @cuDynamicSharedMem T (blockDim().x+2, blockDim().y+2)
    Stencil{N}(dims, buf, args)
end

function load_stencil!(buf, data, i, j, k)
    # translate between local to global indices
    m = threadIdx().x
    n = threadIdx().y
    # halo region 
    m, n = m+1, n+1

    buf[m, n] =  data[i, j, k]

    if m == 2 
        buf[m-1, n] = data[i-1, j, k]
    else m == blockDim().x+1 # boundary of block
        buf[m+1, n] = data[i+1, j, k]
    end
    if n == 2
        buf[m, n-1, k] = data[i, j-1, k]
    else n == blockDim().y+1 # boundary of block
        buf[m, n+1, k] = data[i, j+1, k]
    end
    # NOTE we don't load in the corners yet
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

function Base.iterate(stencil::Stencil{N}) where N
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = 1
    I, J, K = stencil.dims
    if !checkbounds(Bool, CartesianIndices((1:I, 1:J, 1:K)), i,j,k)
        return nothing
    end

    buf = stencil.buf

    regions = ntuple(Val(N)) do ind
        data = stencil.arrays[ind]
        # load the boundary
        m, n = load_stencil!(buf, data, i, j, k-1)
        pre = load_slice(buf, m, n)

        # m, n = load_stencil!(buf, data, i, j, k)
        # current = SMatrix{3,3}(view(buf, m-1:m+1, n-1:m+1))

        # m, n = load_stencil!(buf, data, i, j, k+1)
        # next = SMatrix{3,3}(view(buf, m-1:m+1, n-1:m+1))

        # ldata = cat(pre, current, next, dims=3)
    end

    ((i,j,k,regions...), (regions, k+1))
end


function Base.iterate(stencil::Stencil{N}, (regions, k)) where N
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    I, J, K = stencil.dims
    if !checkbounds(Bool, CartesianIndices((1:I, 1:J, 1:K)), i,j,k)
        return nothing
    end

    buf = stencil.buf

    next_regions = ntuple(Val(N)) do ind
        data = stencil.arrays[ind]
        old = regions[ind]

        m, n = load_stencil!(buf, data, i, j, k+1)
        next = SMatrix{3,3}(view(buf, m-1:m+1, n-1:m+1))
        ldata = cat(old[:,:,2], old[:,:,3], next, dims=3)
    end

    ((i,j,k, next_regions...), (next_regions, k+1))
end

# function kernel(out, U)
#     for (i,j,k, u) in stencil((32,32,32), U) 
#         out[i, j, k] = (u[3, 2, 2] - u[2, 2, 2]) + 0.5 * (u[2, 1, 2] + u[2, 3, 2])
#     end
#     return nothing
# end
# 
# function test(out, U)
#     shmem = 8*8*sizeof(eltype(out))
#     k = GPUifyLoops.contextualize(kernel)
#     @cuda threads=(8,8) blocks=(4,4) k(out, U)
# end
# 
# out = OffsetArray(CuArray(zeros(34,34,34)), 0:33, 0:33, 0:33);
# A = OffsetArray(CuArray(rand(34,34,34)), 0:33, 0:33, 0:33);
