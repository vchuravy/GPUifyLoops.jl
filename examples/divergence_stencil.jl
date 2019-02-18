using GPUifyLoops, BenchmarkTools

# Increment integer with periodic wrapping.
@inline incmod1(a, n) = a == n ? 1 : a+1

# x, y, and z difference operators with periodic boundary conditions.
# Nx, Ny, and Nz are the number of grid points in each dimension.
# They return the difference at grid point (i, j, k).
@inline δx(f, Nx, i, j, k) = @inbounds f[incmod1(i, Nx), j, k] - f[i, j, k]
@inline δy(f, Ny, i, j, k) = @inbounds f[i, incmod1(j, Ny), k] - f[i, j, k]
@inline δz(f, Nz, i, j, k) = @inbounds f[i, j, incmod1(k, Nz)] - f[i, j, k]

# 3D Divergence operator.
@inline div(f, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) = δx(f, Nx, i, j, k) / Δx + δy(f, Ny, i, j, k) / Δy + δz(f, Nz, i, j, k) / Δz

# This is the actual kernel.
function div_kernel(::Val{Dev}, f, div_f) where Dev
    @setup Dev
    
    Nx, Ny, Nz = size(f)
    Δx, Δy, Δz = 1, 1, 1
    
    # Calculate the divergence of f at every point and store it in div_f.
    @loop for k in (1:Nz; blockIdx().z)
        @loop for j in (1:Ny; (blockIdx().y - 1) * blockDim().y + threadIdx().y)
            @loop for i in (1:Nx; (blockIdx().x - 1) * blockDim().x + threadIdx().x)
                @inbounds div_f[i, j, k] = div(f, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
            end
        end
    end
    
    @synchronize
end

# CPU wrapper.
calc_div(f::Array, div_f::Array) = div_kernel(Val(:CPU), f, div_f)

# GPU wrapper.
@static if Base.find_package("CuArrays") !== nothing
    using CuArrays, CUDAnative
    
    @eval function calc_div(f::CuArray, div_f::CuArray)
        Nx, Ny, Nz = size(f)
    
        Tx, Ty = 16, 16  # Threads per block
        Bx, By, Bz = Int(Nx/Tx), Int(Ny/Ty), Nz  # Blocks in grid.
        
        @cuda threads=(Tx, Ty) blocks=(Bx, By, Bz) div_kernel(Val(:GPU), f, div_f)
    end
end


Nx, Ny, Nz = 1024, 1024, 512

xc, yc = rand(Nx, Ny, Nz), rand(Nx, Ny, Nz)
println("CPU:")
display(@benchmark calc_div($xc, $yc))

@static if Base.find_package("CuArrays") !== nothing
    using CuArrays, CUDAnative

    xg, yg = cu(rand(Nx, Ny, Nz)), cu(rand(Nx, Ny, Nz))
    println("GPU:")
    display(@benchmark CuArrays.@sync calc_div($xg, $yg))
end
