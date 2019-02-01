"""
    @prefetch A[1, 2]

Prefetch the memory location accessed by A at location A[1, 1]

TODO:
  - What to do about StructArray 
"""
macro prefetch(expr)
    @assert expr.head == :ref
    A = expr.args[1]
    I = expr.args[2:end]
    esc(quote
        $prefetch($A, $(I...))
    end)
end

Base.@propagate_inbounds function prefetch(A, I...)
    lindex = LinearIndices(A)[I...]
    ptr = pointer(A, lindex)
    __prefetch(ptr, Val(:read), Val(3), Val(:data))
end

@generated function __prefetch(ptr::T, ::Val{RW}, ::Val{Locality}, ::Val{Cache}) where {T, RW, Locality, Cache}
    decls = """
    declare void @llvm.prefetch(i8*, i32, i32, i32)
    """

    if RW == :read
        f_rw = 0
    elseif RW == :write
        f_rw = 1
    end
    
    f_locality = Locality
    
    if Cache == :data
        f_cache = 1
    elseif Cache == :instruction
        f_cache = 0
    end

    ir = """
        %ptr = inttoptr i64 %0 to i8*
        call void @llvm.prefetch(i8* %ptr, i32 $f_rw, i32 $f_locality, i32 $f_cache)
        ret void
    """
    
    quote
        Base.@_inline_meta
        Base.llvmcall(($decls, $ir), Nothing, Tuple{T}, ptr)
    end
end

