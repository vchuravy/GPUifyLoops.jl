using Logging
using CUDAnative
using CuArrays
using GPUifyLoops

function kernel(::Val{3}, ::Val{N}, Q, vgeo, nelem) where N
    DFloat = eltype(Q)
    Nq = N + 1

    @inbounds @loop for e in (1:nelem; blockIdx().x)
        @loop for k in (1:Nq; threadIdx().z)
            @loop for j in (1:Nq; threadIdx().y)
                @loop for i in (1:Nq; threadIdx().x)
                    MJ = vgeo[i,j,k,10,e]
                    ξx, ξy, ξz = vgeo[i,j,k,1,e], vgeo[i,j,k,4,e], vgeo[i,j,k,7,e]
                    ηx, ηy, ηz = vgeo[i,j,k,2,e], vgeo[i,j,k,5,e], vgeo[i,j,k,8,e]
                    ζx, ζy, ζz = vgeo[i,j,k,3,e], vgeo[i,j,k,6,e], vgeo[i,j,k,9,e]
                    z = vgeo[i,j,k,14,e]

                    U, V, W = Q[i,j,k,1,e], Q[i,j,k,2,e], Q[i,j,k,3,e]
                    ρ, E = Q[i,j,k,4,e], Q[i,j,k,5,e]

                    P = 0.4*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*0.9*z)
                end
            end
        end
    end
    nothing
end

function main()
    @info "Starting main"
    N = 4
    nelem = 4000
    DFloat = Float32

    @info "Initializing arrays"
    Nq = N + 1
    Q = CuArray(zeros(DFloat, Nq, Nq, Nq, 5, nelem))
    vgeo = CuArray(zeros(DFloat, Nq, Nq, Nq, 14, nelem))

    @info "Running kernel..."
    @time @launch(CUDA(), threads=(N+1, N+1, N+1), blocks=nelem,
                  kernel(Val(3), Val(N), Q, vgeo, nelem))
    @info "Finished kernel!"

    nothing
end

main()
