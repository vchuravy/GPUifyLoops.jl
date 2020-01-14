@init @require CUDAnative="be33ccc6-a3ff-5ff2-a52e-74243cff1e17" begin
    using .CUDAnative

    @Base.propagate_inbounds function __read_only_load(::CUDA, A::CUDAnative.CuDeviceArray, index::Integer)
        CUDAnative.ldg(A, index)
    end
end

@Base.propagate_inbounds __read_only_load(::CPU, A::Array, index::Integer) = A[index]

"""
    read_only_load(A, i)

Index the array `A` with the linear index `i`.  On the GPU this uses the
read-only texture cache (e.g., via `ldg` in CUDAnative) so the memory `A`
refers to must not be written to during the kernel execution.

See also: `Base.getindex`, `CUDAnative.ldg`
"""
@Base.propagate_inbounds read_only_load(A, index) = __read_only_load(backend(), A, index)
