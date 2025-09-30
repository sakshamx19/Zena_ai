from Utilis.functions import extract_api_version
from langchain_openai import ChatOpenAI,AzureChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  
from langchain_huggingface.llms import HuggingFacePipeline  
from langchain_ollama import OllamaLLM




class LLMModule:
    def __init__(self, model_type=None, model_name=None, api_key=None, endpoint_url=None, temperature=0.7, max_tokens=150, top_p=0.9):
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.client = self.test_connection()


    def test_connection(self):
        # Implement a method to test the connection to the LLM API
        if self.model_type.lower() == "openai":
            import openai
            client = openai.Client(api_key=self.api_key)
            try:
                response = client.chats.create(model=self.model_name, messages=[{"role": "user", "content": "Hello"}])
            except Exception as e:
                print(f"Error: {e}")
            return response
        
        elif self.model_type.lower() == "azure":
            
            client = AzureChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                api_version=extract_api_version(self.endpoint_url),
                azure_endpoint=self.endpoint_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
            try:
                response = client.invoke("Hello")
                return client
            except Exception as e:
                print(f"Error: {e}")
                return 
            
        elif self.model_type.lower() == "huggingface":
            model_id = self.model_name
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForCausalLM.from_pretrained(model_id)
            gen = pipeline("text-generation", model=mdl, tokenizer=tok,
                           max_new_tokens=self.max_tokens, pad_token_id=tok.eos_token_id, temperature=self.temperature, top_p=self.top_p)

            client = HuggingFacePipeline(pipeline=gen)
            try:
                response = client.invoke("Hello")
                print(response)
                return client
            except Exception as e:
                print(f"Error: {e}")
                return
        elif self.model_type.lower() == "ollama":

            client = OllamaLLM(model=self.model_name)
            try:
                response = client.invoke("Hello")
                print(response)
                return client
            except Exception as e:
                print(f"Error: {e}")
                return
    
    
    
    
    
    def generate_text(self, prompt):
        # Implement a method to generate text based on the given prompt
        pass

    def set_parameters(self, temperature=None, max_tokens=None, top_p=None):
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if top_p is not None:
            self.top_p = top_p