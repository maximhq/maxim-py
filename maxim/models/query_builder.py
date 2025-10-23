from dataclasses import dataclass
from typing import Dict, Literal, Union
import json

class QueryRuleType(str):
    DeploymentVar = "deploymentVar"
    Tag = "tag"


@dataclass
class QueryRule():
    query: str
    operator: Literal["AND", "OR"]
    exact_match: bool
    scopes: Dict[str, str]


class QueryBuilder:
    """
    This class represents a query builder. Users can use this class to build a query rule for fetching prompts, agents or workflow from Maxim server.
    """

    def __init__(self):
        self.query: str = ""
        self.scopes: Dict[str, str] = {}
        self.operator: Literal["AND", "OR"] = "AND"
        self.is_exact_match: bool = False

    def and_(self) -> 'QueryBuilder':
        """
        Sets the operator for combining query rules to 'AND'.

        Returns:
            QueryBuilder: The current QueryBuilder instance for method chaining.
        """
        self.operator = "AND"
        return self

    def or_(self) -> 'QueryBuilder':
        """
        Sets the operator for combining query rules to 'OR'.

        Returns:
            QueryBuilder: The current QueryBuilder instance for method chaining.
        """
        self.operator = "OR"
        return self

    def folder(self, folderId: str) -> 'QueryBuilder':
        """
        Sets the folder scope for the query.

        Args:
            folderId (str): The ID of the folder to set as the scope.

        Returns:
            QueryBuilder: The current QueryBuilder instance for method chaining.
        """
        self.scopes["folder"] = folderId
        return self

    def exact_match(self) -> 'QueryBuilder':
        """
        Sets the exact match flag to True.

        Returns:
            QueryBuilder: The current QueryBuilder instance for method chaining.
        """
        self.is_exact_match = True
        return self

    def deployment_var(self, key: str, value: Union[str, int, bool, list], enforce: bool = True) -> 'QueryBuilder':
        """
        Adds a deployment variable rule to the query.

        Args:
            key (str): The key of the deployment variable.
            value (Union[str, int, bool, list]): The value of the deployment variable.
            enforce (bool, optional): Whether to enforce the deployment variable. Defaults to True.

        Returns:
            QueryBuilder: The current QueryBuilder instance for method chaining.
        """

        if len(self.query) > 0:
            self.query += ","
        self.query += f"{'!!' if enforce else ''}{key}={json.dumps(value) if isinstance(value, (list, bool)) else value}"
        return self

    def tag(self, key: str, value: Union[str, int, bool], enforce: bool = False) -> 'QueryBuilder':
        """
        Adds a tag rule to the query.

        Args:
            key (str): The key of the tag.
            value (Union[str, int, bool]): The value of the tag.
            enforce (bool, optional): Whether to enforce the tag. Defaults to False.

        Returns:
            QueryBuilder: The current QueryBuilder instance for method chaining.
        """
        if len(self.query) > 0:
            self.query += ","
        self.query += f"{'!!' if enforce else ''}{key}={value}"
        return self

    def prompt_version_number(self, number: int) -> 'QueryBuilder':
        """
        Adds a rule to fetch a specific prompt version by its numeric version.

        Args:
            number (int): The version number of the prompt to fetch.

        Returns:
            QueryBuilder: The current QueryBuilder instance for method chaining.
        """
        if len(self.query) > 0:
            self.query += ","
        self.query += f"promptVersionNumber={number}"
        return self

    def build(self) -> QueryRule:
        """
        Builds the final query rule.

        Raises:
            ValueError: If the query is empty after trimming.

        Returns:
            QueryRule: A QueryRule instance with the built query.
        """
        if len(self.query.strip()) == 0:
            raise ValueError("Cannot build an empty query. Please add at least one rule (deploymentVar or tag).")
        return QueryRule(
            query=self.query,
            operator=self.operator,
            exact_match=self.is_exact_match,
            scopes=self.scopes
        )
