import logging
import os
import unittest
from dotenv import load_dotenv

from maxim import Maxim
from maxim.models import QueryBuilder
load_dotenv()

apiKey = os.getenv("MAXIM_API_KEY")
promptId = os.getenv("MAXIM_PROMPT_1_ID")
promptVersionId = os.getenv("PROMPT_VERSION_ID")
folderID = os.getenv("FOLDER_ID")
baseUrl = os.getenv("MAXIM_BASE_URL") or "https://app.getmaxim.ai"


class TestMaximPromptManagement(unittest.TestCase):
    def setUp(self):
        # Clear singleton instance if it exists
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        self.maxim = Maxim(
            {
                "api_key": apiKey,
                "debug": True,
                "prompt_management": True,
                "base_url": baseUrl
            }
        )

    def tearDown(self):
        # Clean up the Maxim instance
        if hasattr(self, "maxim"):
            self.maxim.cleanup()

        # Clear singleton instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

    def test_get_prompt_with_deployment_variables(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder().and_().deployment_var("Environment", "Prod").build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.version_id, os.getenv("MAXIM_PROMPT_1_VERSION_1_ID"))
        self.assertEqual(prompt.model, "gpt-4o")
        self.assertEqual(prompt.provider, "openai")
        self.assertEqual(prompt.messages[0].content, "You are a helpful assistant. You talk like Chandler from Friends.")
        self.assertEqual(len(prompt.messages), 1)
        try:
            resp = prompt.run(
                "What is Cosmos about?",
            )
        except Exception as e:
            self.fail(f"prompt.run() raised an exception: {e}")

    def test_getPrompt_with_multiselect_deployment_variables_and_execute(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "Prod")
            .deployment_var("tenant_id", ["1"])
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")

        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.model, "gpt-4o")
        self.assertEqual(prompt.provider, "openai")
        self.assertEqual(
            prompt.messages[0].content,
            "You are a helpful assistant. Answer the user query.",
        )
        self.assertEqual(len(prompt.messages), 1)
        try:
            resp = prompt.run(
                "What is Cosmos about?",
            )
        except Exception as e:
            self.fail(f"prompt.run() raised an exception: {e}")

    def test_custom_prompt_execution(self) -> None:
        resp = self.maxim.chat_completion(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "who is sachin tendulkar"},
            ],
            temperature=0.8,
        )
        print(resp)

    def test_getPrompt_with_deployment_variables_Environment_prod_and_TenantId_123(
        self,
    ):
        prompt_3_id = os.getenv("MAXIM_PROMPT_3_ID")
        prompt = self.maxim.get_prompt(
            prompt_3_id,
            QueryBuilder()
            .and_()
            .deployment_var("Test number", 123)
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, prompt_3_id)
        self.assertEqual(prompt.version_id, os.getenv("MAXIM_PROMPT_3_VERSION_2_ID"))
        self.assertEqual(len(prompt.messages), 1)

    def test_getPrompt_with_deployment_variables_multiselect(
        self,
    ):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var(
                "Test multi select", ["tenant1", "tenant2", "tenant3", "tenant4"]
            )
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.version_id, os.getenv("MAXIM_PROMPT_1_VERSION_4_ID"))
        self.assertEqual(len(prompt.messages), 1)

    def test_getPrompt_with_deployment_variables_multiselect_includes(
        self,
    ):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder().and_().deployment_var("Test multi select", ["tenant1"]).build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.version_id, os.getenv("MAXIM_PROMPT_1_VERSION_5_ID"))
        res = prompt.run("What is Cosmos about?")
        print(res.choices[0].message.content)

    def test_if_prompt_cache_works_fine(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "Prod")
            .deployment_var("Tenants", ["tenant1"])
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        prompt2 = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "Prod")
            .deployment_var("Tenants", ["tenant1"])
            .build(),
        )
        if prompt2 is None:
            raise Exception("Prompt2 not found")
        self.assertEqual(prompt2.prompt_id, promptId)

    def test_if_fallback_works_fine(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "Prod")
            .deployment_var("TenantId", 1234, False)
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)

    def test_if_fallback_works_fine_forceful(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "Prod")
            .deployment_var("tenant_id", "1")
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        logging.debug(prompt.prompt_id)
        self.assertEqual(prompt.prompt_id, promptId)

    @unittest.skip("Skipping test that uses excluded testAndTagsCustomerIdGradeAndTest")
    def test_fetch_prompts_using_tags(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "test")
            .tag("CustomerId", 1234)
            .tag("grade", "A")
            .tag("test", True)
            .exact_match()
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)

    def test_fetch_all_prompts_deployed_on_prod(self):
        prompts = self.maxim.get_prompts(
            QueryBuilder().and_().deployment_var("Environment", "Prod").build(),
        )
        print([p.version_id for p in prompts])

    # # Issue : Doesn't work properly
    @unittest.skip(
        "Skipping test that uses excluded prodPromptsWithOptionalCustomerId1234"
    )
    def test_fetch_all_prompts_deployed_on_prod_with_tag_filters(self):
        prompts = self.maxim.get_prompts(
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "Prod")
            .tag("tenant_id", "1")
            .build(),
        )
        print(f"prompts : {[p for p in prompts]}")

    def test_getFolderUsingId(self):
        folder = self.maxim.get_folder_by_id(os.getenv("MAXIM_FOLDER_1_ID"))
        if folder is None:
            raise Exception("Folder not found")
        self.assertEqual(folder.name, "SDK Tests")

    def test_getFolderUsingTags(self):
        folders = self.maxim.get_folders(
            QueryBuilder().and_().tag("Testing", True).build()
        )
        self.assertEqual(folders[0].name, "SDK Tests")
        self.assertEqual(len(folders), 1)

    # Issue : When only environment Var is given it is not able to retriev
    @unittest.skip(
        "Skipping test that uses excluded testFolderId and testFolderEnvStageTenant123PromptVersion"
    )
    def test_getPromptsFromAFolder(self):
        prompts = self.maxim.get_prompts(
            QueryBuilder()
            .and_()
            .folder(os.getenv("MAXIM_FOLDER_2_ID"))
            .deployment_var("Environment", "Prod")
            .build(),
        )
        self.assertEqual(len(prompts), 1)

if __name__ == "__main__":
    unittest.main()
