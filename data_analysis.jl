using CSV
using DataFrames
using StatsBase, Statistics
using Distributions

data_nofault = readtable("Documents/Stanford/aa228/finalproject/Field Snaps - No warranty examples.txt", separator = ';')


for (i, n) in enumerate(names(data))
    println(rpad(i,5), rpad(n, 30), sum(ismissing.(data[n])))
end

function remove_missing_cols!(data)
    for n in reverse(names(data))
        if sum(ismissing.(data[n])) > 0
            delete!(data, n)
        end
    end
end
missing_cols!(data_nofault)
disallowmissing!(data_nofault)


mean_squared_error(datavec, D::Type{<:Distribution}, nbins = 20) = mean_squared_error(fit(Histogram, datavec), fit(D, datavec), nbins)
function mean_squared_error(h::Histogram, D::Distribution, nbins = 20)
    bins = collect(h.edges[1])
    wts = normalize(h.weights)

    err = 0.0
    for i in 1:length(bins)-1

        x = (bins[i] + bins[i+1]) /2

        fit_y = pdf(D, x)
        data_y = wts[i]

        err += (fit_y - data_y)^2
    end

    err/(length(bins)-1)
end



