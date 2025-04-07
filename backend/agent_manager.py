from langchain.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, Tool, AgentType
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

def load_llm():
    model_name = "meta-llama/Llama-2-7b-chat-hf"  
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return HuggingFacePipeline(pipeline=pipe)

def run_agent(query, document_text):
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
