from openai import OpenAI, AsyncOpenAI
import json

class BaseExtractior:
    """
        A class to handle any extraction using OpenAI's GPT models.
    """

    def __init__(self, openai_client: OpenAI|AsyncOpenAI) -> None:
        """
            Initializes the BaseExtractior with an OpenAI client.

            Args:
                openai_client (OpenAI | AsyncOpenAI): An instance of OpenAI or AsyncOpenAI client to interact with the API.
        """
        self.openai_client = openai_client
        self.is_async = False
        if isinstance(self.openai_client, AsyncOpenAI):
            self.is_async = True

    def _get_completion_result(self,
                                model,
                                messages: list[dict],
                                temperature: float
    ) -> str:

        completion = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        return completion.choices[0].message.content
    

    async def _async_get_completion_result(self,
                                model,
                                messages: list[dict],
                                temperature: float
    ) -> str:

        completion = await self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )

        return completion.choices[0].message.content



    def _get_completion_parsed_result(self,
                                    model,
                                    messages: list[dict],
                                    response_format,
                                    temperature: float,
                                    key: str):

        completion = self.openai_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )

        return json.loads(completion.choices[0].message.content)[key]
    
    async def _async_get_completion_parsed_result(self,
                                            model,
                                            messages: list[dict],
                                            response_format,
                                            temperature: float,
                                            key: str):

        completion = await self.openai_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
        )

        return json.loads(completion.choices[0].message.content)[key]