#!/usr/bin/env julia
#
# Manifolds.jl side of the cross-language comparison.
#
# Reads the shared fixtures, computes exp / log / dist / parallel transport,
# and writes both the resulting values and the timings. Values come first: a
# speed comparison between implementations that disagree is meaningless.
#
# Convention mapping, as recorded in make_fixtures.py:
#   fixture sphere dim N  ->  Manifolds.Sphere(N-1)          (ambient N)
#   fixture spd    dim N  ->  SymmetricPositiveDefinite(N)   (affine-invariant)
#
# Run:
#   JULIA_DEPOT_PATH=/home/julia/depot julia --project=. bench_geometry.jl
#
using BenchmarkTools
using JSON
using LinearAlgebra
using Manifolds
using Statistics

const ROOT = joinpath(@__DIR__, "..")
const FIXTURES = joinpath(ROOT, "fixtures", "geometry_cases.json")
const OUT = joinpath(ROOT, "results", "julia_geometry.jsonl")

"""
Reconstruct a square matrix from the row-major nested list the fixtures hold.

JSON gives a vector of row vectors; `reduce(hcat, ...)` on those produces the
transpose, so this transposes back. Every fixture matrix here is symmetric, so
a mistake would be invisible in the values and would only show up as an
asymmetry error later. Being explicit is cheaper than debugging that.
"""
matrix_from(rows) = permutedims(reduce(hcat, [Float64.(r) for r in rows]))

vector_from(xs) = Float64.(xs)

"""
Median and interquartile range of a benchmark trial, in nanoseconds.

BenchmarkTools drives its own sample count and warmup, so the numbers are
directly comparable to criterion's rather than to a hand-rolled timing loop.
"""
function timing_ns(trial)
    ts = trial.times
    (median = median(ts), q1 = quantile(ts, 0.25), q3 = quantile(ts, 0.75))
end

function run_case(case)
    kind = case["manifold"]
    dim = case["dim"]
    records = []

    if kind == "sphere"
        # Manifolds.jl Sphere(n) is the unit sphere in R^(n+1).
        M = Sphere(dim - 1)
        p = vector_from(case["p"])
        v = vector_from(case["v"])
        q = vector_from(case["q"])
    elseif kind == "spd"
        M = SymmetricPositiveDefinite(dim)
        p = matrix_from(case["p"])
        v = matrix_from(case["v"])
        q = matrix_from(case["q"])
    else
        error("unknown manifold $kind")
    end

    # Values, for the agreement comparison.
    exp_val = exp(M, p, v)
    log_val = log(M, p, q)
    dist_val = distance(M, p, q)

    # Timings.
    t_exp = timing_ns(@benchmark exp($M, $p, $v))
    t_log = timing_ns(@benchmark log($M, $p, $q))
    t_dist = timing_ns(@benchmark distance($M, $p, $q))

    push!(records, Dict(
        "lib" => "manifolds.jl", "manifold" => kind, "dim" => dim, "op" => "exp",
        "value" => vec(exp_val),
        "median_ns" => t_exp.median, "q1_ns" => t_exp.q1, "q3_ns" => t_exp.q3,
    ))
    push!(records, Dict(
        "lib" => "manifolds.jl", "manifold" => kind, "dim" => dim, "op" => "log",
        "value" => vec(log_val),
        "median_ns" => t_log.median, "q1_ns" => t_log.q1, "q3_ns" => t_log.q3,
    ))
    push!(records, Dict(
        "lib" => "manifolds.jl", "manifold" => kind, "dim" => dim, "op" => "dist",
        "value" => [dist_val],
        "median_ns" => t_dist.median, "q1_ns" => t_dist.q1, "q3_ns" => t_dist.q3,
    ))

    # Parallel transport is only compared on the sphere: the SPD transport
    # convention differs between libraries (along the geodesic versus to a
    # point with a chosen frame), so a mismatch there would measure the
    # convention rather than the implementation.
    if kind == "sphere"
        pt_val = parallel_transport_to(M, p, v, q)
        t_pt = timing_ns(@benchmark parallel_transport_to($M, $p, $v, $q))
        push!(records, Dict(
            "lib" => "manifolds.jl", "manifold" => kind, "dim" => dim, "op" => "transport",
            "value" => vec(pt_val),
            "median_ns" => t_pt.median, "q1_ns" => t_pt.q1, "q3_ns" => t_pt.q3,
        ))
    end

    records
end

function main()
    data = JSON.parsefile(FIXTURES)
    mkpath(dirname(OUT))

    all_records = []
    for case in data["cases"]
        @info "benchmarking" manifold=case["manifold"] dim=case["dim"]
        append!(all_records, run_case(case))
    end

    open(OUT, "w") do io
        for r in all_records
            println(io, JSON.json(r))
        end
    end

    @info "wrote records" n=length(all_records) path=OUT
end

main()
