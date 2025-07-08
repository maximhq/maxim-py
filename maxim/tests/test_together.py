import unittest
import os
import dotenv
from together import Together
from maxim import Maxim
from maxim.logger.together import MaximTogetherClient
from maxim.maxim import ConfigDict

dotenv.load_dotenv()

together_api_key = os.getenv("TOGETHER_API_KEY")
maxim_api_key = os.getenv("MAXIM_API_KEY")
maxim_base_url = os.getenv("MAXIM_BASE_URL") or "http://localhost:3000"
maxim_log_repo_id = os.getenv("MAXIM_LOG_REPO_ID")

logger = Maxim(config=ConfigDict(
    base_url=maxim_base_url,
    api_key=maxim_api_key,
    log_repo_id=maxim_log_repo_id,
)).logger()

together_client = Together(api_key=together_api_key)
maxim_client = MaximTogetherClient(together_client, logger=logger)

# response = maxim_client.completions.create(
#     prompt="What is the history of F1?",
#     model="Qwen/Qwen2.5-7B-Instruct-Turbo",
# )

response = maxim_client.chat.completions.create(
    messages=[
        {"role": "user", "content": "What is the history of F1?"}
    ],
    model="Qwen/Qwen2.5-7B-Instruct-Turbo",
)

logger.flush()

print(response)
    

# class TestTogether(unittest.TestCase):
#     def setUp(self):
#         # Reset Maxim instance to ensure clean state
#         if hasattr(Maxim, "_instance"):
#             delattr(Maxim, "_instance")
#         self.logger = Maxim(config=ConfigDict(
#             base_url=maxim_base_url,
#             api_key=maxim_api_key,
#             log_repo_id=maxim_log_repo_id,
#         )).logger()
#
#     def test_completions_using_wrapper(self):
#         client = MaximTogetherClient(
#             Together(api_key=together_api_key),
#             logger=self.logger
#         )
#         response = client.completions.create(
#             prompt="What is the history of F1?",
#             model="Qwen/Qwen2.5-7B-Instruct-Turbo",
#         )
#         print(response)
#         
#         # Verify response structure
#         # self.assertIsNotNone(response)
#         # self.assertTrue(hasattr(response, "choices"))
#         # self.assertTrue(len(response.choices) > 0)
#         
#         # Verify logging
#         self.logger.flush()
#         # self.mock_writer.assert_entity_action_count("trace", "create", 1)
#         # self.mock_writer.assert_entity_action_count("trace", "add-generation", 1)
#         # self.mock_writer.assert_entity_action_count("generation", "result", 1)
#         # self.mock_writer.assert_entity_action_count("trace", "end", 1)
#
#     def tearDown(self) -> None:
#         return super().tearDown()
#
# if __name__ == "__main__":
#     unittest.main()