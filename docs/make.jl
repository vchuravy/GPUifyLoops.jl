using Documenter, GPUifyLoops

makedocs(
    modules = [GPUifyLoops],
    format = :html,
    sitename = "GPUifyLoops.jl",
    pages = [
        "Home"    => "index.md",
    ],
    doctest = true
)

deploydocs(
    repo = "github.com/vchuravy/GPUifyLoops.jl.git",
    julia = "",
    osname = "",
    # no need to build anything here, re-use output of `makedocs`
    target = "build",
    deps = nothing,
    make = nothing
)
