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

