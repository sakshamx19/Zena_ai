from LLM.module import LLMModule

def test_llm_module():
    # Unified prompts
    system_prompt = "You are a helpful assistant named ZENA_AI."
    user_prompt = "What is the capital of France?"

    # Test OpenAI connection
    print("Testing OpenAI connection...")
    openai_module = LLMModule(
        model_type="openai",
        model_name="gpt-3.5-turbo",
        api_key="your_openai_api_key",
        system_prompt=system_prompt
    )
    if openai_module.client:
        print("OpenAI connection successful!")
        try:
            print(f"Sending user prompt: {user_prompt}")
            response = openai_module.generate_text(user_prompt)
            print(f"Generated Response (OpenAI): {response}")
        except Exception as e:
            print(f"Error generating response (OpenAI): {e}")
    else:
        print("OpenAI connection failed.")

    # Test Azure connection
    print("\nTesting Azure connection...")
    azure_module = LLMModule(
        model_type="azure",
        model_name="pssl-gpt-4o",
        api_key="d004ba2610a04317bda192df2e53b71c",
        endpoint_url="https://genai-pssl-sweden.openai.azure.com/openai/deployments/pssl-gpt-4o/chat/completions?api-version=2024-08-01-preview",
        temperature=0.2
    )
    if azure_module.client:
        print("Azure connection successful")
        try:
            azure_module.set_system_prompt(system_prompt)
            print(f"Sending user prompt: {user_prompt}")
            response = azure_module.generate_text(user_prompt)
            print(f"Generated Response (Azure): {response}")
        except Exception as e:
            print(f"Error generating response (Azure): {e}")
    else:
        print("Azure connection failed.")

    # Test HuggingFace connection
    print("\nTesting HuggingFace connection...")
    huggingface_module = LLMModule(
        model_type="huggingface",
        model_name="gpt2",
        system_prompt=system_prompt
    )
    if huggingface_module.client:
        print("HuggingFace connection successful!")
        try:
            print(f"Sending user prompt: {user_prompt}")
            response = huggingface_module.generate_text(user_prompt)
            print(f"Generated Response (HuggingFace): {response}")
        except Exception as e:
            print(f"Error generating response (HuggingFace): {e}")
    else:
        print("HuggingFace connection failed.")

    # Test Ollama connection
    print("\nTesting Ollama connection...")
    ollama_module = LLMModule(
        model_type="ollama",
        model_name="ollama-model",
        system_prompt=system_prompt
    )
    if ollama_module.client:
        print("Ollama connection successful!")
        try:
            print(f"Sending user prompt: {user_prompt}")
            response = ollama_module.generate_text(user_prompt)
            print(f"Generated Response (Ollama): {response}")
        except Exception as e:
            print(f"Error generating response (Ollama): {e}")
    else:
        print("Ollama connection failed.")

    # Test Azure OpenAI connection with configurable system prompt
    print("\nTesting Azure OpenAI connection with configurable system prompt...")
    azure_openai_module = LLMModule(
        model_type="azure",
        model_name="pssl-gpt-4o",
        api_key="d004ba2610a04317bda192df2e53b71c",
        endpoint_url="https://genai-pssl-sweden.openai.azure.com/openai/deployments/pssl-gpt-4o/chat/completions?api-version=2024-02-15-preview",
        temperature=0.2
    )

    user_prompt = "What is the capital of France?"
    print(f"Sending user prompt: {user_prompt}")

    if azure_openai_module.client:
        print("Azure OpenAI connection successful!")
        try:
            azure_openai_module.set_system_prompt(system_prompt)
            # Print raw Azure OpenAI response as well
            raw_messages = [
                {"role": "system", "content": azure_openai_module.system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ]
            raw_response = azure_openai_module.client.invoke(raw_messages)
            print(f"Raw Azure OpenAI response: {raw_response}")

            response = azure_openai_module.generate_text(user_prompt)
            if response:
                print(f"Generated Response (Azure, system+user): {response}")
            else:
                print("No response generated")
        except Exception as e:
            print(f"Error generating response: {e}")
    else:
        print("Azure OpenAI connection failed.")

if __name__ == "__main__":
    print("Starting test script...")
    try:
        test_llm_module()
    except Exception as e:
        print(f"Test script failed with error: {e}")


