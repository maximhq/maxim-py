import concurrent.futures
import json
import logging
import os
from time import sleep

from flask import Flask, jsonify
from langchain_openai import AzureChatOpenAI

from maxim import Config, Maxim
from maxim.logger import LoggerConfig
from maxim.logger.langchain import MaximLangchainTracer

logging.basicConfig(level=logging.INFO)


app = Flask(__name__)

with open(str(f"{os.getcwd()}/libs/maxim-py/maxim/tests/testConfig.json")) as f:
    data = json.load(f)

env = "dev"

azureOpenAIBaseUrl = data["azureOpenAIBaseUrl"]
azureOpenAIKey = data["azureOpenAIKey"]
openAIKey = data["openAIKey"]
apiKey = data[env]["apiKey"]
baseUrl = data[env]["baseUrl"]
repoId = data[env]["repoId"]

maxim = Maxim(config=Config(api_key=apiKey, base_url=baseUrl, debug=True))
logger = maxim.logger(LoggerConfig(id=repoId))


def call_llm_1():
    maxim = Maxim(config=Config(api_key=apiKey, base_url=baseUrl, debug=True))
    logger = maxim.logger(LoggerConfig(id=repoId))
    model = AzureChatOpenAI(
        api_key=azureOpenAIKey,
        model="gpt-35-turbo-16k",
        azure_endpoint=azureOpenAIBaseUrl,
        callbacks=[MaximLangchainTracer(logger)],
        api_version="2024-02-01",
    )
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "How are you doing."),
    ]
    result = model.invoke(messages)
    return result.content


def call_llm_2():
    maxim = Maxim(config=Config(api_key=apiKey, base_url=baseUrl, debug=True))
    logger = maxim.logger(LoggerConfig(id=repoId))
    model = AzureChatOpenAI(
        api_key=azureOpenAIKey,
        model="gpt-35-turbo-16k",
        azure_endpoint=azureOpenAIBaseUrl,
        callbacks=[MaximLangchainTracer(logger)],
        api_version="2024-02-01",
    )
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "How are you doing."),
    ]
    result = model.invoke(messages)
    print(result)
    sleep(10)
    return result.content


@app.post("/test")
def eval():
    with concurrent.futures.ThreadPoolExecutor() as executor:
        print("submitting")
        future1 = executor.submit(call_llm_1)
        future2 = executor.submit(call_llm_2)
        completed, not_completed = concurrent.futures.wait(
            [future1, future2], 20, return_when=concurrent.futures.FIRST_COMPLETED
        )
        print("completed", completed, "not_completed", not_completed)
        response1 = None
        response2 = None
        for future in completed:
            response1 = future.result()
        completed, _ = concurrent.futures.wait(not_completed, 20)
        for future in completed:
            print("retrying", not_completed)
            response2 = future.result()
    return jsonify({"llm1": response1, "llm2": response2})


if __name__ == "__main__":
    app.run()
