from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field
import json
from langchain_openai import AzureChatOpenAI


class Person(BaseModel):
    name: str = Field(description="Full name of the person")
    profession: str = Field(description="Profession or occupation")
    birth_year: int = Field(description="Year of birth")


class PydanticOutputParser(BaseOutputParser):
    def __init__(self, schema: type[BaseModel]):
        object.__setattr__(self, "schema", schema)

    def parse(self, text: str | dict | BaseModel) -> dict:
        if isinstance(text, self.schema):
            obj = text
        elif isinstance(text, dict):
            obj = self.schema(**text)
        elif isinstance(text, str):
            obj = self.schema(**json.loads(text))
        else:
            raise ValueError(f"Unexpected type: {type(text)}")
        return obj.model_dump()

llm = AzureChatOpenAI(
    deployment_name="pssl-gpt-4o",
    api_key="d004ba2610a04317bda192df2e53b71c",
    api_version="2024-08-01-preview",
    azure_endpoint="https://genai-pssl-sweden.openai.azure.com/",
    temperature=0,
)

structured_llm = llm.with_structured_output(Person, method="function_calling")

parser = PydanticOutputParser(Person)

result = structured_llm.invoke("Give me details about Mahatma Gandhi.")
parsed_output = parser.parse(result)

print(parsed_output)
