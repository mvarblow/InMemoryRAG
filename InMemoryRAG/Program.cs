using Azure;
using Azure.AI.Inference;
using InMemoryRAG;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Connectors.InMemory;
using System;
using System.Collections.Generic;
using ChatRole = Microsoft.Extensions.AI.ChatRole;

// RAG chat settings:
bool showSearchResults = false;
Uri endpoint = new("https://models.inference.ai.azure.com");
string vectorEmbeddingsModel = "text-embedding-3-small";
string chatModel = "Phi-3.5-MoE-instruct";

string? githubToken = Environment.GetEnvironmentVariable("GITHUB_TOKEN");
if (string.IsNullOrEmpty(githubToken))
{
    var config = new ConfigurationBuilder()
        .AddUserSecrets<Program>() // This requires the above using directive and package reference  
        .Build();
    githubToken = config["GITHUB_TOKEN"];
}
while (string.IsNullOrEmpty(githubToken))
{
    Console.Write("Please enter a valid github token: ");
    githubToken = Console.ReadLine();
}

AzureKeyCredential credential = new(githubToken); // githubToken is retrieved from the environment variables

var movieData = new List<Movie>
{
    new Movie { Key = 1, Title = "The Matrix", Description = "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers." },
    new Movie { Key = 2, Title = "Inception", Description = "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O." },
    new Movie { Key = 3, Title = "Interstellar", Description = "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival." }
};

// Create an in-memory vector database with a collection for the movies
var vectorStore = new InMemoryVectorStore();
var movies = vectorStore.GetCollection<int, Movie>("movies");
await movies.CreateCollectionIfNotExistsAsync();

// Create the embedding generator (to convert movie descriptions to vector representations using our embeddings model)
IEmbeddingGenerator<string, Embedding<float>> generator = 
    new EmbeddingsClient(endpoint, credential)
        .AsIEmbeddingGenerator(vectorEmbeddingsModel);

// Use the embegging generator to populate the vector property and add each movie to the in-memory vector database
foreach (var movie in movieData)
{
    movie.Vector = await generator.GenerateEmbeddingVectorAsync(movie.Description);
    await movies.UpsertAsync(movie);
}

// Create the chat client using our selected LLM
IChatClient chatClient = new ChatCompletionsClient(endpoint, credential)
    .AsIChatClient(chatModel);

// Start the chat session and continue until the user stops providing new input
Console.WriteLine("Movie search assistant. Starting chat session...");
Console.Write("User> ");
string? query = Console.ReadLine();
while (!string.IsNullOrEmpty(query))
{
    Console.WriteLine();

    // search the knowledge store based on the user's prompt
    ReadOnlyMemory<float> queryEmbedding = await generator.GenerateEmbeddingVectorAsync(query);
    VectorSearchResults<Movie> searchResults = await movies.VectorizedSearchAsync(
        queryEmbedding, 
        new VectorSearchOptions<Movie> { Top = 2 });

    if (showSearchResults)
    {
        // let's see the results just so we know what they look like
        Console.WriteLine("-----------------------------------------------------------------");
        Console.WriteLine("Vector search results:");
        Console.WriteLine();
        await foreach (var result in searchResults.Results)
        {
            Console.WriteLine($"Title: {result.Record.Title}");
            Console.WriteLine($"Description: {result.Record.Description}");
            Console.WriteLine($"Score: {result.Score}");
            Console.WriteLine();
        }
        Console.WriteLine("-----------------------------------------------------------------");
        Console.WriteLine();
    }

    List<ChatMessage> conversation =
    [
        new ChatMessage(ChatRole.System, "You are a friendly assistant who helps me select a movie to watch based my preferences. Only recommend movies I tell you about."),
        new ChatMessage(ChatRole.User, query)
    ];

    // add the search results to the conversation
    await foreach (var result in searchResults.Results)
    {
        conversation.Add(new ChatMessage(ChatRole.User, $"The movie \"{result.Record.Title}\" is about: {result.Record.Description}"));
    }

    // send the conversation to the model
    var response = await chatClient.GetResponseAsync(conversation);

    // add the assistant message to the conversation
    foreach (var message in response.Messages)
    {
        conversation.Add(new ChatMessage(message.Role, message.Text));
        Console.WriteLine($"Bot> {message.Text}");
    }

    Console.WriteLine();
    Console.Write("User> ");
    query = Console.ReadLine();
}
