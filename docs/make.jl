using Documenter, GPUifyLoops

makedocs(
    modules = [GPUifyLoops],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    sitename = "GPUifyLoops.jl",
    pages = [
        "Home"    => "index.md",
    ],
    doctest = true
)

