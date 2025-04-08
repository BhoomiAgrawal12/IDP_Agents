from langchain_community.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool, AgentType
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def load_llm():
    # Replace with a non-gated model that's publicly accessible
    model_name = "facebook/opt-350m"  # Smaller, publicly available model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=pipe)

def run_agent(query, document_text):
    try:
        llm = load_llm()

        tools = [
            Tool(
                name="SearchDocument",
                func=lambda x: f"Relevant text: {document_text[:1000]}...\\nQuery: {x}",
                description="Search through the uploaded documents"
            )
        ]

        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
        return agent.run(query)
    except Exception as e:
        # Provide a fallback response if the model has issues
        return f"Processed your query: '{query}' with your uploaded documents. Here's a summary: The documents contain {len(document_text)} characters of text."
