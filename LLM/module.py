from Utilis.functions import extract_api_version
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_ollama import OllamaLLM
from typing_extensions import override  # Import the @override decorator


class BaseConnection:
    def __init__(self, model_name, api_key=None, endpoint_url=None, temperature=None, max_tokens=None, top_p=None):
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.client = None  

    def connect(self):
        raise NotImplementedError("Subclasses must implement the connect method")


class OpenAIConnection(BaseConnection):
    @override
    def connect(self):
        import openai
        self.client = openai.Client(api_key=self.api_key)  
        try:
            response = self.client.chats.create(model=self.model_name, messages=[{"role": "user", "content": "Hello"}])
            return self.client
        except Exception as e:
            print(f"Error: {e}")
            return None


class AzureConnection(BaseConnection):
    @override
    def connect(self):
        self.client = AzureChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            api_version=extract_api_version(self.endpoint_url),
            azure_endpoint=self.endpoint_url,
            temperature=self.temperature or 0.7,
            max_tokens=self.max_tokens or 150,
            top_p=self.top_p or 0.9
        )
        try:
            response = self.client.invoke("Hello")
            return self.client
        except Exception as e:
            print(f"Error: {e}")
            return None


class HuggingFaceConnection(BaseConnection):
    @override
    def connect(self):
        model_id = self.model_name
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id)
        gen = pipeline("text-generation", model=mdl, tokenizer=tok,
                       max_new_tokens=self.max_tokens or 150,
                       pad_token_id=tok.eos_token_id,
                       temperature=self.temperature or 0.7,
                       top_p=self.top_p or 0.9)
        self.client = HuggingFacePipeline(pipeline=gen)  # Set the client attribute
        try:
            response = self.client.invoke("Hello")
            print(response)
            return self.client
        except Exception as e:
            print(f"Error: {e}")
            return None


class OllamaConnection(BaseConnection):
    @override
    def connect(self):
        self.client = OllamaLLM(model=self.model_name)  # Set the client attribute
        try:
            response = self.client.invoke("Hello")
            print(response)
            return self.client
        except Exception as e:
            print(f"Error: {e}")
            return None


class LLMModule:
    def __init__(self, model_type=None, model_name=None, api_key=None, endpoint_url=None, temperature=None, max_tokens=None, top_p=None):
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.client = self.test_connection()

    def test_connection(self):
        connection_classes = {
            "openai": OpenAIConnection,
            "azure": AzureConnection,
            "huggingface": HuggingFaceConnection,
            "ollama": OllamaConnection
        }
        connection_class = connection_classes.get(self.model_type.lower())
        if connection_class:
            connection = connection_class(
                model_name=self.model_name,
                api_key=self.api_key,
                endpoint_url=self.endpoint_url,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
            return connection.connect()
        else:
            print(f"Error: Unsupported model type '{self.model_type}'")
            return None

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