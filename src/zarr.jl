using Zarr: AbstractStore, Zarr, zopen, zgroup, zcreate
export DiskArrayShuffleStore
struct DiskArrayShuffleStore{S<:DiskArrayShuffler} <: AbstractStore
    s::S
    name::String
    metadata::Dict{String,Vector{UInt8}}
end
Base.show(io::IO,s::DiskArrayShuffleStore) = print(io,"Shuffled Store")
function DiskArrayShuffleStore(s::DiskArrayShuffler; name = "batches", attrs = Dict(),nbatches=100)
    onear = s.currentcoll[].chunkviews |> first |> first
    soutput = size(onear)
    cs = (soutput..., s.sampler.batchsize)
    soutput = (cs...,nbatches)
    #Create a mock Zarr array to contain the metadata
    g = zgroup(Zarr.DictStore())
    zcreate(eltype(onear),g,name,soutput..., compressor=Zarr.NoCompressor(), attrs=attrs)
    Zarr.consolidate_metadata(g)
    DiskArrayShuffleStore(s,name,g.storage.a)
end



function Base.getindex(s::DiskArrayShuffleStore,i::AbstractString)
    if i in (s.name * "/.zarray", s.name*"/.zattrs",".zgroup",".zmetadata")
        return get(s.metadata,i,nothing)
    else
        schunk = split(i,'/')[end]
        chunk = parse.(Int,split(schunk,'.'))
        all(iszero,chunk[2:end]) || return nothing
        r = readbatch(s.s)
        nd = ndims(first(r))
        rone = reduce((i,j)->cat(i,j,dims=nd+1),r)
        return collect(vec(reinterpret(UInt8,rone)))
    end
end
