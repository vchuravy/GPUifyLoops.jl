using Adapt, CUDAnative, CuArrays, OffsetArrays, GPUifyLoops
using BenchmarkTools

# Adapt an offset CuArray to work nicely with CUDA kernels.
Adapt.adapt_structure(to, x::OffsetArray) = OffsetArray(adapt(to, parent(x)), x.offsets)

const N = 512
const Nx, Ny, Nz = N, N, N
const Δx, Δy, Δz = 1, 1, 1

# Difference operators on a staggered grid.
@inline δx(i, j, k, f) = @inbounds f[i,   j, k] - f[i-1, j, k]
@inline δy(i, j, k, f) = @inbounds f[i, j,   k] - f[i, j-1, k]
@inline δz(i, j, k, f) = @inbounds f[i, j,   k] - f[i, j, k-1]

@inline δx²(i, j, k, f) = δx(i+1, j,     k, f) - δx(i, j, k, f)
@inline δy²(i, j, k, f) = δy(i,   j+1,   k, f) - δy(i, j, k, f)
@inline δz²(i, j, k, f) = δz(i,   j,   k+1, f) - δz(i, j, k, f)

@inline ∇²(i, j, k, u, v, w) = δx²(i, j, k, u) / Δx^2 + δy²(i, j, k, v) / Δy^2 + δz²(i, j, k, w) / Δz^2

function laplacian!(u, v, w, ∇)
    @loop for k in (1:Nz; (blockIdx().z - 1) * blockDim().z + threadIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds ∇[i, j, k] = ∇²(i, j, k, u, v, w)
            end
        end
    end
end

u  = rand(Nx+2, Ny+2, Nz+2) |> CuArray
v  = rand(Nx+2, Ny+2, Nz+2) |> CuArray
w  = rand(Nx+2, Ny+2, Nz+2) |> CuArray
∇ = zeros(Nx+2, Ny+2, Nz+2) |> CuArray

u = OffsetArray(u, 0:Nx+1, 0:Ny+1, 0:Nz+1)
v = OffsetArray(v, 0:Nx+1, 0:Ny+1, 0:Nz+1)
w = OffsetArray(w, 0:Nx+1, 0:Ny+1, 0:Nz+1)
∇ = OffsetArray(∇, 0:Nx+1, 0:Ny+1, 0:Nz+1)

T = floor(Int, ∛N)
B = floor(Int, N / T)

@benchmark CuArrays.@sync @launch CUDA() threads=(T, T, T) blocks=(B, B, B) laplacian!(u, v, w, ∇)

function laplacian_stencil!(u, v, w, ∇)
    for (i, j, k, uₛ, vₛ, wₛ) in stencil((Nx, Ny, Nz), u, v, w)
        @inbounds ∇[i, j, k] = ∇²(2, 2, 2, u, v, w)
    end
end

@benchmark CuArrays.@sync @launch CUDA() threads=(T, T, T) blocks=(B, B, B) laplacian_stencil!(u, v, w, ∇)

