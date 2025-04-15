using Microsoft.Extensions.VectorData;
using System;

namespace InMemoryRAG;

public class Movie
{
    [VectorStoreRecordKey]
    public int Key { get; set; }

    [VectorStoreRecordData]
    public required string Title { get; set; }

    [VectorStoreRecordData]
    public required string Description { get; set; }

    [VectorStoreRecordVector(384, DistanceFunction.CosineSimilarity)]
    public ReadOnlyMemory<float> Vector { get; set; }
}
