import json
import logging
import os
import unittest

from maxim import Maxim
from maxim.models import QueryBuilder

# reading testConfig.json and setting the values

# Get the directory where this test file is located
with open(str(f"{os.getcwd()}/maxim/tests/testConfig.json")) as f:
    data = json.load(f)

# local config
env = "prod"
apiKey = data[env]["apiKey"]
promptId = data[env]["promptId"]
promptVersionId = data[env]["promptVersionId"]
baseUrl = data[env]["baseUrl"]
folderID = data[env]["folderId"]


class TestMaximPromptManagement(unittest.TestCase):
    def setUp(self):
        # Clear singleton instance if it exists
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

        self.maxim = Maxim(
            {
                "api_key": apiKey,
                "base_url": baseUrl,
                "debug": True,
                "prompt_management": True,
            }
        )

    def tearDown(self):
        # Clean up the Maxim instance
        if hasattr(self, "maxim"):
            self.maxim.cleanup()

        # Clear singleton instance
        if hasattr(Maxim, "_instance"):
            delattr(Maxim, "_instance")

    def test_getPrompt_with_deployment_variables_and_execute(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder().and_().deployment_var("Environment", "prod").build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")

        print(f"Provider: {prompt.provider}")

        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.model, "gpt-4")
        self.assertEqual(prompt.provider, "openai")
        self.assertEqual(prompt.version_id, promptVersionId)
        # self.assertEqual(prompt.messages[0].content, "You are a helpful assistant")
        # self.assertEqual(len(prompt.messages), 1)
        resp = prompt.run(
            "who is sachin tendulkar",
        )
        print(
            f">>>RESPONSE: {resp.choices[0].message.content if resp else 'No response'}"
        )

    def test_run_hosted_prompt_with_vision_model(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder().and_().deployment_var("Environment", "stage").build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        resp = prompt.run(
            "explain this image",
            image_urls=[
                {
                    "url": "https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8aW1hZ2V8ZW58MHx8MHx8fDA%3D",
                    "detail": "auto",
                }
            ],
        )
        print(resp)

    def test_getPrompt_with_deployment_variables_Environment_prod(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder().and_().deployment_var("Environment", "prod").build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.version_id, data[env]["prodPromptVersionId"])
        self.assertEqual(prompt.model, "gpt-4")
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
            .deployment_var("Environment", "prod")
            .deployment_var("TenantId", 123)
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.version_id, data[env]["prodAndT123PromptVersionId"])
        self.assertEqual(len(prompt.messages), 2)

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
            .deployment_var("TenantId", 123)
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.version_id, data[env]["prodAndT123PromptVersionId"])
        prompt2 = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "prod")
            .deployment_var("TenantId", 123)
            .build(),
        )
        if prompt2 is None:
            raise Exception("Prompt2 not found")
        self.assertEqual(prompt2.prompt_id, promptId)
        self.assertEqual(prompt2.version_id, data[env]["prodAndT123PromptVersionId"])

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
        self.assertEqual(prompt.version_id, data[env]["prodPromptVersionId"])

    def test_if_fallback_works_fine_forceful(self):
        prompt = self.maxim.get_prompt(
            promptId,
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "prod")
            .deployment_var("TenantId", 123)
            .build(),
        )
        if prompt is None:
            raise Exception("Prompt not found")
        logging.debug(prompt.prompt_id)
        self.assertEqual(prompt.prompt_id, promptId)
        self.assertEqual(prompt.version_id, data[env]["prodAndT123PromptVersionId"])

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
        self.assertEqual(
            prompt.version_id, data[env]["testAndTagsCustomerIdGradeAndTest"]
        )

    def test_fetch_all_prompts_deployed_on_prod(self):
        prompts = self.maxim.get_prompts(
            QueryBuilder().and_().deployment_var("Environment", "prod").build(),
        )
        print([p.version_id for p in prompts])
        for p in prompts:
            if p is not None:
                self.assertTrue(p.version_id in data[env]["prodPromptVersions"])
        self.assertEqual(len(prompts), len(data[env]["prodPromptVersions"]))

    # # Issue : Doesn't work properly
    @unittest.skip(
        "Skipping test that uses excluded prodPromptsWithOptionalCustomerId1234"
    )
    def test_fetch_all_prompts_deployed_on_prod_with_tag_filters(self):
        prompts = self.maxim.get_prompts(
            QueryBuilder()
            .and_()
            .deployment_var("Environment", "prod")
            .tag("CustomerId", 1234, False)
            .build(),
        )
        print(f"prompts : {[p for p in prompts]}")
        for p in prompts:
            if p is not None:
                self.assertIn(
                    p.version_id, data[env]["prodPromptsWithOptionalCustomerId1234"]
                )

    def test_getFolderUsingId(self):
        folder = self.maxim.get_folder_by_id(folderID)
        if folder is None:
            raise Exception("Folder not found")
        self.assertEqual(folder.name, "Test Folder")

    def test_getFolderUsingTags(self):
        folders = self.maxim.get_folders(
            QueryBuilder().and_().tag("test", True).build()
        )
        self.assertEqual(folders[0].name, "Test Folder")
        self.assertEqual(len(folders), 1)

    # Issue : When only environment Var is given it is not able to retriev
    @unittest.skip(
        "Skipping test that uses excluded testFolderId and testFolderEnvStageTenant123PromptVersion"
    )
    def test_getPromptsFromAFolder(self):
        prompts = self.maxim.get_prompts(
            QueryBuilder()
            .and_()
            .folder(data[env]["testFolderId"])
            .deployment_var("Environment", "stage")
            .deployment_var("TenantId", 123)
            .build(),
        )
        self.assertEqual(len(prompts), 1)
        self.assertEqual(
            prompts[0].version_id,
            data[env]["testFolderEnvStageTenant123PromptVersion"],
        )

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
