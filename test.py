from LLM.module import LLMModule

def test_llm_module():
    # Test OpenAI connection
    print("Testing OpenAI connection...")
    openai_module = LLMModule(
        model_type="openai",
        model_name="gpt-3.5-turbo",
        api_key="your_openai_api_key"
    )
    if openai_module.client:
        print("OpenAI connection successful!")
    else:
        print("OpenAI connection failed.")

    # Test Azure connection
    print("\nTesting Azure connection...")
    azure_module = LLMModule(
        model_type="azure",
        model_name="pssl-gpt-4o",
        api_key="d004ba2610a04317bda192df2e53b71c",
        endpoint_url="https://genai-pssl-sweden.openai.azure.com/openai/deployments/pssl-gpt-4o/chat/completions?api-version=2024-08-01-preview",
        temperature = 0.2
    )
    if azure_module.client:
        print("Azure connection successful!")
    else:
        print("Azure connection failed.")

    # Test HuggingFace connection
    print("\nTesting HuggingFace connection...")
    huggingface_module = LLMModule(
        model_type="huggingface",
        model_name="gpt2"
    )
    if huggingface_module.client:
        print("HuggingFace connection successful!")
    else:
        print("HuggingFace connection failed.")

    # Test Ollama connection
    print("\nTesting Ollama connection...")
    ollama_module = LLMModule(
        model_type="ollama",
        model_name="ollama-model"
    )
    if ollama_module.client:
        print("Ollama connection successful!")
    else:
        print("Ollama connection failed.")

if __name__ == "__main__":
    test_llm_module()
