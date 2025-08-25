import json
import logging
import os
import unittest

from maxim import Maxim
from maxim.models import QueryBuilder

# reading testConfig.json and setting the values

env = "dev"
apiKey = os.getenv("MAXIM_API_KEY")
promptId = os.getenv("PROMPT_ID")
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
            QueryBuilder().and_().deployment_var("Environment", "prod").build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.version_id, promptVersionId)
        self.assertEqual(prompt.model, "gpt-3.5-turbo")
        self.assertEqual(prompt.provider, "openai")
        self.assertEqual(prompt.messages[0].content, "You are a helpful assistant")
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
            .deployment_var("Environment", "prod")
            .deployment_var("tenant_id", ["1"])
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")

        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.model, "gpt-3.5-turbo")
        self.assertEqual(prompt.provider, "openai")
        self.assertEqual(
            prompt.messages[0].content,
            "You are a helpful assistant. And you talk like chandler",
        )
        self.assertEqual(len(prompt.messages), 1)
        try:
            resp = prompt.run(
                "What is Cosmos about?",
            )
        except Exception as e:
            self.fail(f"prompt.run() raised an exception: {e}")

    def test_prompt_with_vision_model(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder().and_().deployment_var("Environment", "stage").build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.model, "anthropic.claude-3-5-sonnet-20240620-v1:0")
        self.assertEqual(prompt.provider, "bedrock")
        self.assertEqual(
            prompt.messages[0].content,
            "You are a helpful assistant. \n\nBe polite in every case",
        )

    def test_getPrompt_with_deployment_variables_Environment_prod(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder().and_().deployment_var("Environment", "prod").build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.model, "gpt-3.5-turbo")
        self.assertEqual(prompt.provider, "openai")
        # Message content may vary, so we just check that messages exist
        self.assertTrue(len(prompt.messages) > 0)

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
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "Prod")
            .deployment_var("TenantId", 123)
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.version_id, data[env]["prodAndT123PromptVersionId"])
        self.assertEqual(len(prompt.messages), 2)

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
        self.assertEqual(prompt.version_id, data[env]["prodAndT123PromptVersionId"])
        self.assertEqual(len(prompt.messages), 2)

    def test_getPrompt_with_deployment_variables_multiselect_includes(
        self,
    ):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder().and_().deployment_var("Tenants", ["tenant1"]).build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.version_id, data[env]["promptVersionId"])
        res = prompt.run("What is Cosmos about?")
        print(res.choices[0].message.content)

    def test_getPrompt_with_deployment_variables_Environment_stage_and_TenantId_123(
        self,
    ):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "beta")
            .deployment_var("TenantId", 123)
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.version_id, data[env]["stageAndT123PromptVersionId"])
        self.assertEqual(len(prompt.messages), 1)

    def test_if_prompt_cache_works_fine(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "prod")
            .deployment_var("tenant_id", "1")
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        prompt2 = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "prod")
            .deployment_var("tenant_id", "1")
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
            .deployment_var("Environment", "prod")
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
            .deployment_var("Environment", "prod")
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
            QueryBuilder().and_().deployment_var("Environment", "prod").build(),
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
            .deployment_var("Environment", "prod")
            .tag("tenant_id", "1")
            .build(),
        )
        print(f"prompts : {[p for p in prompts]}")

    def test_getFolderUsingId(self):
        folder = self.maxim.get_folder_by_id(folderID)
        if folder is None:
            raise Exception("Folder not found")
        self.assertEqual(folder.name, "SDK Tests")

    def test_getFolderUsingTags(self):
        folders = self.maxim.get_folders(
            QueryBuilder().and_().tag("test", True).build()
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
            .folder(folderID)
            .deployment_var("Environment", "prod")
            .build(),
        )
        self.assertEqual(len(prompts), 1)

    def test_add_dataset_entries(self):
        self.skipTest("")
        self.maxim.addDatasetEntries(data[env]["datasetIdToAddData"], [self.payload])

    def test_addDatasetEntriesShouldThrowErrorOnInvalidDatasetId(self):
        self.skipTest("")
        with self.assertRaises(Exception) as context:
            self.maxim.addDatasetEntries(
                data[env]["fakeDataSetIdToThrowError"], [self.payload]
            )
        self.assertEqual(str(context.exception), "Error: 404 - Not Found")


if __name__ == "__main__":
    unittest.main()
