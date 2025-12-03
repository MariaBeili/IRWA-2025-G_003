import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class RAGGenerator:
    # IMPROVEMENT 1: Better Prompt Engineering with Persona and constraints
    PROMPT_TEMPLATE = """
    You are an expert personal shopper assistant for a fashion e-commerce site.
    
    User Query: "{user_query}"
    
    Here are the top products found (Format: ID | Name | Price | Rating | Description):
    {retrieved_results}
    
    INSTRUCTIONS:
    1. Analyze the products above.
    2. Recommend the best option based on the user's query.
    3. Explain WHY it is the best choice (mention price, fabric, brand, or specific features).
    4. If there is a good alternative (e.g., cheaper or different style), mention it.
    5. Keep the tone helpful and professional.
    6. If NO products match the query well, say exactly: "I couldn't find any products that perfectly match your request."
    
    Output your response in clear, concise paragraphs. Do not use Markdown formatting like **bold** or # Headers, just plain text is fine.
    """

    def generate_response(self, user_query: str, retrieved_results: list, top_N: int = 5) -> str:
        """
        Generates a summary/recommendation using an LLM.
        """
        # IMPROVEMENT 3: Safety check for API key
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return "RAG System Disabled: No API Key found."

        try:
            client = Groq(api_key=api_key)
            model_name = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

            # IMPROVEMENT 2: Enriched Context (Sending specific metadata like Price/Rating)
            # We limit to top_N (e.g., 5) to save tokens and focus on best results
            context_list = []
            for doc in retrieved_results[:top_N]:
                # Handle dictionary or object access safely
                pid = getattr(doc, 'pid', doc.get('pid', 'N/A'))
                title = getattr(doc, 'title', doc.get('title', 'N/A'))
                price = getattr(doc, 'selling_price', doc.get('selling_price', 'N/A'))
                rating = getattr(doc, 'average_rating', doc.get('average_rating', 'N/A'))
                desc = getattr(doc, 'description', doc.get('description', ''))[:150] # Truncate desc
                
                context_list.append(f"{pid} | {title} | {price} | {rating} | {desc}...")

            formatted_results = "\n".join(context_list)

            prompt = self.PROMPT_TEMPLATE.format(
                user_query=user_query,
                retrieved_results=formatted_results
            )

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful shopping assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=model_name,
                temperature=0.5, # Lower temperature for more factual answers
            )

            return chat_completion.choices[0].message.content

        except Exception as e:
            print(f"RAG Error: {e}")
            return "I'm sorry, I couldn't generate a summary at this moment."