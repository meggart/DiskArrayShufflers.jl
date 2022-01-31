module DiskArrayShufflers
# using Zarr
# using Zarr: AbstractStore
using StatsBase
include("pickaxisarray.jl")
using .PickAxisArrays
using DiskArrays: GridChunks, DiskArrays
export DiskArrayShuffler, readbatch, FullSliceSampler


function getbatchinds(batchsize,currentchunk_views)
    s = eachindex.(currentchunk_views)
    wchunks = weights(length.(currentchunk_views))
    map(1:batchsize) do _
        chunk = sample(1:length(s),wchunks)
        ii = sample(s[chunk])
        chunk,ii
    end
end


struct ChunkCollection{P<:PickAxisArray,B}
    chunkviews::Vector{P}
    batches::B
end
retdims(::ChunkCollection{<:PickAxisArray{T}}) where T = ndims(T)
function getbatch(c::ChunkCollection)
    cv = c.chunkviews
    bnext = nextbatch(c.batches)
    if bnext === nothing
        return nothing
    else
        return [cv[ii[1]][ii[2]] for ii in bnext]
    end
end


struct BatchIndexList{I}
    indices::I
    icur::Ref{Int}
    lock::ReentrantLock
end
function nextbatch(b::BatchIndexList) 
    lock(b.lock) do
        icur = b.icur[] 
        if icur > length(b.indices)
            nothing 
        else
            b.icur[] = icur+1
            b.indices[icur]
        end
    end
end

struct FullSliceSampler{SD}
    batchsize::Int
    nchunks::Int
    batchespercoll::Int
    sd::Val{SD}
end
function FullSliceSampler(sd::Int...;batchsize=100, nchunks=10,batchespercoll=100)
    FullSliceSampler(batchsize,nchunks,batchespercoll,Val(sd))
end

function create_new_chunklist(s::FullSliceSampler{SD}, a, c::GridChunks) where SD
    chunkinds = sample(c,s.nchunks,replace=false)
    data = [a[ci...] for ci in chunkinds]
    isinsampledim = [i ∉ SD ? true : Colon() for i in 1:ndims(a)]
    data_views = PickAxisArray.(data, Ref(isinsampledim)); 
    bi = [getbatchinds(s.batchsize,data_views) for _ in 1:s.batchespercoll]
    bl = BatchIndexList(bi,Ref(1),ReentrantLock())
    ChunkCollection(data_views, bl)
end



struct DiskArrayShuffler{A<:AbstractArray,G<:GridChunks,CT,S}
    parent::A
    parentchunks::G
    currentcoll::Ref{CT}
    nextcoll::Ref{Task}
    sampler::S
    lock::ReentrantLock
end
function DiskArrayShuffler(parent, sampler; chunks::GridChunks = DiskArrays.eachchunk(parent))
    coll = create_new_chunklist(sampler, parent, chunks)
    next = @async create_new_chunklist(sampler, parent, chunks)
    DiskArrayShuffler(parent,chunks,Ref(coll),Ref(next),sampler,ReentrantLock())
end
function readbatch(s::DiskArrayShuffler)
    r = getbatch(s.currentcoll[])
    if r !== nothing
        return r
    else
        if trylock(s.lock)
            @debug "Exchanging chunk collection"
            s.currentcoll[] = fetch(s.nextcoll[])
            s.nextcoll[] = @async create_new_chunklist(s.sampler, s.parent, s.parentchunks)
            unlock(s.lock)
        else
            sleep(0.2)
        end
        return readbatch(s)
    end
end
include("zarr.jl")

end
