using Adapt, CUDAnative, CuArrays, OffsetArrays, GPUifyLoops
using Test, BenchmarkTools

using GPUifyLoops: stencil, Full

# Adapt an offset CuArray to work nicely with CUDA kernels.
Adapt.adapt_structure(to, x::OffsetArray) = OffsetArray(adapt(to, parent(x)), x.offsets)

const N = 32
const Nx, Ny, Nz = N, N, N
const Δx, Δy, Δz = 1, 1, 1

u = ones(Nx+2, Ny+2, Nz+2) |> CuArray
v = 2 .* ones(Nx+2, Ny+2, Nz+2) |> CuArray
w = 4 .* ones(Nx+2, Ny+2, Nz+2) |> CuArray
S = zeros(Nx+2, Ny+2, Nz+2) |> CuArray

u = OffsetArray(u, 0:Nx+1, 0:Ny+1, 0:Nz+1)
v = OffsetArray(v, 0:Nx+1, 0:Ny+1, 0:Nz+1)
w = OffsetArray(w, 0:Nx+1, 0:Ny+1, 0:Nz+1)
S = OffsetArray(S, 0:Nx+1, 0:Ny+1, 0:Nz+1)

T = min(16, floor(Int, √N))
B = ceil(Int, N / T)

function sum_stencil!(u, v, w, S)
    for (i, j, k, uₛ, vₛ, wₛ) in stencil((Nx, Ny, Nz), Full(), u, v, w)
        @inbounds S[i, j, k] = uₛ[2, 2, 2] + vₛ[2, 2, 2] + wₛ[2, 2, 2]
    end
end

@launch CUDA() threads=(T, T, 1) blocks=(B, B, 1) shmem=((T+2)*(T+2)*sizeof(Float64)) sum_stencil!(u, v, w, S)

@test all(S[1:Nx, 1:Ny, 1:Nz] .≈ 7)

