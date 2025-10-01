from Utilis.functions import extract_api_version
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_ollama import OllamaLLM
from typing_extensions import override


class BaseConnection:
    def __init__(self, model_name, model_type, api_key=None, endpoint_url=None, temperature=None, max_tokens=None, top_p=None):
        self.model_name = model_name
        self.model_type = model_type
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
        self.client = HuggingFacePipeline(pipeline=gen)
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
        self.client = OllamaLLM(model=self.model_name)
        try:
            response = self.client.invoke("Hello")
            print(response)
            return self.client
        except Exception as e:
            print(f"Error: {e}")
            return None


class LLMModule:
    def __init__(self, model_type=None, model_name=None, api_key=None, endpoint_url=None, temperature=None, max_tokens=None, top_p=None, system_prompt=None):
        self.model_type = model_type
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.system_prompt = system_prompt  # Configurable system prompt
        self.client = self.test_connection()

    def set_system_prompt(self, system_prompt):
        """
        Set or update the system prompt.
        """
        self.system_prompt = system_prompt
        print(f"System prompt updated to: {self.system_prompt}")

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
                model_type=self.model_type,
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

    def generate_text(self, user_prompt):
        print(f"Generating text for prompt: {user_prompt}")
        if not self.client:
            print("Error: No client initialized. Please check the connection.")
            return None

        try:
            if self.model_type.lower() == "azure":
                messages = [
                    {"role": "system", "content": self.system_prompt or "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}
                ]
                print(f"Sending messages to Azure: {messages}")
                response = self.client.invoke(messages)
                print(f"Azure response received: {response}")
                return response.content if hasattr(response, 'content') else response

            # Example for HuggingFace-like models
            elif self.model_type.lower() == "huggingface":
                combined_prompt = f"{self.system_prompt}\n{user_prompt}"
                response = self.client.invoke(combined_prompt)
                return response

            # Example for Ollama-like models
            elif self.model_type.lower() == "ollama":
                combined_prompt = f"{self.system_prompt}\n{user_prompt}"
                response = self.client.invoke(combined_prompt)
                return response

            else:
                print(f"Error: Unsupported model type '{self.model_type}'")
                return None

        except Exception as e:
            print(f"Error during text generation: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def set_parameters(self, temperature=None, max_tokens=None, top_p=None):
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if top_p is not None:
            self.top_p = top_p