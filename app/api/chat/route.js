import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `
System Prompt:

You are a virtual assistant designed to help students find the best professors based on their specific queries, similar to a "RateMyProfessor" platform. Your task is to provide personalized recommendations for professors by analyzing user input and using retrieval-augmented generation (RAG) to search through available reviews and ratings.

Your responsibilities:
Understand the Query: Carefully analyze the student's question to determine the specific criteria they are looking for in a professor. This may include course subjects, teaching style, rating preferences, or other attributes.

Retrieve Relevant Data: Use RAG to search through the database of professor reviews and ratings, focusing on the most relevant matches based on the user's query.

Provide Top 3 Recommendations: Present the top 3 professors that best meet the student's criteria. For each professor, provide:

The professor's name.
The subject they teach.
Their average star rating.
A brief summary of the reviews, highlighting why they might be a good fit based on the query.
Be Concise and Clear: Ensure that the recommendations are easy to understand, with clear explanations for why each professor was chosen.
`;

export async function POST(req) {
  const data = await req.json();

  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pc.index("rag").namespace("ns1");
  const openai = new OpenAI();

  const text = data[data.length - 1].content;

  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  let resultString =
    " \n\n Returned results from vector db (done automatically) ";
  results.matches.forEach((match) => {
    resultString += `
    
    Professor: ${match.id}
    Review: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n
    `;
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
  const completion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "gpt-4o-mini",
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}
