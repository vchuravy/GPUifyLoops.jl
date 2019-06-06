GPUifyLoops.jl
==============
*Support for writing loop-based code that executes both on CPU and GPU*

[![][docs-latest-img]][docs-latest-url]

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://juliagpu.gitlab.io/GPUifyLoops.jl/

Installation
------------

GPUifyLoops is a registered package, and can be installed using the Julia package
manager.

```julia
julia>]
(v1.1) pkg> add GPUifyLoops
```

**Note**: The current version of this package requires Julia 1.1.

Development
-----------

In order to test this package locally you need to do:

```
julia --project=test/gpuenv
julia> ]
(gpuenv) pkg> resolve
(gpuenv) pkg> instantiate
```

This will resolve the GPU environment, please do not checking changes to `test/gpuenv/`.
Then you can run the tests with `julia --project=test/gpuenv test/runtests.jl`

License
-------

GPUifyLoops.jl is licensed under [MIT license](LICENSE.md).
