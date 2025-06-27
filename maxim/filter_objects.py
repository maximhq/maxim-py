from typing import List, Optional
import re
import json

from .models import RuleGroupType, RuleType


class IncomingQuery:
    """
    Represents an incoming query with its components.

    Attributes:
        query (str): The query string.
        operator (str): The operator used in the query.
        exactMatch (bool): Indicates whether the query requires an exact match.
    """

    def __init__(self, query: str, operator: str, exactMatch: bool):
        """
        Initialize an IncomingQuery object.

        Args:
            query (str): The query string.
            operator (str): The operator used in the query.
            exactMatch (bool): Indicates whether the query requires an exact match.
        """
        self.query = query
        self.operator = operator
        self.exactMatch = exactMatch


class QueryObject:
    """
    Represents a query object with its components.

    Attributes:
        id (str): The ID of the query object.
        query (RuleGroupType): The query object.
    """

    def __init__(self, id: str, query: RuleGroupType):
        self.id = id
        self.query = query


def parse_incoming_query(incoming_query: str) -> List[RuleType]:
    """
    Parses an incoming query string into a list of RuleType objects.

    Args:
        incomingQuery (str): The incoming query string.

    Returns:
        List[RuleType]: A list of RuleType objects.
    """
    if not incoming_query.strip():
        return []
    operators = ["!=", ">=", "<=", ">", "<", "includes", "does not include", "="]
    result = []
    # Only considers commas that are not inside square brackets
    split_query = re.split(r",(?![^\[\]]*\])", incoming_query)
    for condition in split_query:
        no_operator_found = True
        for op in operators:
            if op in condition:
                no_operator_found = False
                field, value = map(str.strip, condition.split(op))
                exact_match = False
                if field.startswith("!!"):
                    exact_match = True
                    field = field[2:]
                parsed = False
                # converting 'True'/'False':str to True/False:bool
                if value == "True" or value == "False":
                    value = bool(value)
                    parsed = True
                if not parsed:
                    if type(value) == bool:
                        value = bool(value)
                        parsed = True
                    elif type(value) is int:
                        value = int(value)
                        parsed = True
                if not parsed:
                    # Checks value for string array
                    try:
                        parsed_value = json.loads(value)
                        if (
                            isinstance(parsed_value, list)
                            and len(parsed_value) > 0
                            and isinstance(parsed_value[0], str)
                        ):
                            value = parsed_value
                    except (json.JSONDecodeError, ValueError):
                        pass
                result.append(
                    RuleType(
                        field=field, value=value, operator=op, exactMatch=exact_match
                    )
                )
        if no_operator_found:
            raise ValueError(f'Unsupported operator found in condition "{condition}"')
    return result


def evaluate_rule_group(
    rule_group: RuleGroupType, incoming_query_rules: List[RuleType]
) -> bool:
    """
    Evaluates a rule group against incoming query rules.

    Args:
        ruleGroup (RuleGroupType): The rule group to evaluate.
        incomingQueryRules (List[RuleType]): The incoming query rules.

    Returns:
        bool: True if the rule group matches, False otherwise.
    """
    matched_rules = []
    match_results = []
    for rule in rule_group.rules:
        if isinstance(rule, RuleGroupType):
            match_results.append(evaluate_rule_group(rule, incoming_query_rules))
        else:
            match = False
            for incomingRule in incoming_query_rules:
                if rule.field == incomingRule.field and condition_met(
                    rule, incomingRule
                ):
                    match = match or True
                    matched_rules.append(incomingRule)
            match_results.append(match)
            if match:
                matched_rules.append(rule)
    exact_matches = all(
        rule.exactMatch == False or rule in matched_rules
        for rule in incoming_query_rules
    )
    if not exact_matches:
        return False
    if rule_group.combinator == "AND":
        return all(match_results)
    else:
        return any(match_results)


# Had to write this extensive logic in python since type support in python isn't as extensive as typescript


def check_operator_match(field_rule: RuleType, field_incoming_rule: RuleType) -> bool:
    """
    Checks if the operator between two rules matches.

    Args:
        fieldRule (RuleType): The rule with the operator.
        fieldIncomingRule (RuleType): The incoming rule.

    Returns:
        bool: True if the operator matches, False otherwise.
    """
    operator = field_rule.operator

    if operator == "=":
        if isinstance(field_rule.value, list):
            return json.dumps(field_rule.value) == json.dumps(field_incoming_rule.value)
        return field_rule.value == field_incoming_rule.value

    if operator == "!=":
        return field_rule.value != field_incoming_rule.value

    if operator == ">":
        if isinstance(field_rule.value, int) and isinstance(
            field_incoming_rule.value, int
        ):
            return field_rule.value > field_incoming_rule.value
        if isinstance(field_rule.value, str) and isinstance(
            field_incoming_rule.value, str
        ):
            return field_rule.value > field_incoming_rule.value
        raise ValueError(
            f"Cannot operate {operator} on types {field_rule} and {field_incoming_rule}"
        )

    if operator == "<":
        if isinstance(field_rule.value, int) and isinstance(
            field_incoming_rule.value, int
        ):
            return field_rule.value < field_incoming_rule.value
        if isinstance(field_rule.value, str) and isinstance(
            field_incoming_rule.value, str
        ):
            return field_rule.value < field_incoming_rule.value
        raise ValueError(
            f"Cannot operate {operator} on types {field_rule} and {field_incoming_rule}"
        )

    if operator == ">=":
        if isinstance(field_rule.value, int) and isinstance(
            field_incoming_rule.value, int
        ):
            return field_rule.value >= field_incoming_rule.value
        if isinstance(field_rule.value, str) and isinstance(
            field_incoming_rule.value, str
        ):
            return field_rule.value >= field_incoming_rule.value

        raise ValueError(
            f"Cannot operate {operator} on types {field_rule} and {field_incoming_rule}"
        )

    if operator == "<=":
        if isinstance(field_rule.value, int) and isinstance(
            field_incoming_rule.value, int
        ):
            return field_rule.value <= field_incoming_rule.value
        if isinstance(field_rule.value, str) and isinstance(
            field_incoming_rule.value, str
        ):
            return field_rule.value <= field_incoming_rule.value

        raise ValueError(
            f"Cannot operate {operator} on types {field_rule} and {field_incoming_rule}"
        )

    if operator == "includes":
        if isinstance(field_rule.value, str) and isinstance(
            field_incoming_rule.value, str
        ):
            return field_incoming_rule.value in field_rule.value
        if isinstance(field_rule.value, list) and isinstance(
            field_incoming_rule.value, list
        ):
            return all(el in field_rule.value for el in field_incoming_rule.value)
        if isinstance(field_rule.value, list) and isinstance(
            field_incoming_rule.value, str
        ):
            return field_incoming_rule.value in field_rule.value
        if isinstance(field_rule.value, str) and isinstance(
            field_incoming_rule.value, list
        ):
            return field_rule.value in field_incoming_rule.value
        raise ValueError(
            f"Cannot operate {operator} on types {field_rule} and {field_incoming_rule}"
        )

    if operator == "does not include":
        if isinstance(field_rule.value, str) and isinstance(
            field_incoming_rule.value, str
        ):
            return field_incoming_rule.value not in field_rule.value
        if isinstance(field_rule.value, list) and isinstance(
            field_incoming_rule.value, list
        ):
            return field_incoming_rule.value not in field_rule.value
        if isinstance(field_rule.value, list) and isinstance(
            field_incoming_rule.value, str
        ):
            return field_incoming_rule.value not in field_rule.value
        if isinstance(field_rule.value, str) and isinstance(
            field_incoming_rule.value, list
        ):
            return field_rule.value not in field_incoming_rule.value
        raise ValueError(
            f"Cannot operate {operator} on types {field_rule} and {field_incoming_rule}"
        )

    return False


def condition_met(field_rule: RuleType, field_incoming_rule: RuleType) -> bool:
    """
    Checks if the condition between two rules is met.

    Args:
        fieldRule (RuleType): The rule with the condition.
        fieldIncomingRule (RuleType): The incoming rule.

    Returns:
        bool: True if the condition is met, False otherwise.
    """
    if not isinstance(field_rule.value, type(field_incoming_rule.value)):
        if isinstance(field_rule.value, int):
            if isinstance(field_incoming_rule.value, int) or isinstance(
                field_incoming_rule.value, bool
            ):
                field_incoming_rule.value = int(field_incoming_rule.value)

        # revisit : possible error : everything will be true with bool(fieldIncomingRule.value)
        elif isinstance(field_rule.value, bool):
            field_incoming_rule.value = bool(field_incoming_rule.value)
        elif isinstance(field_rule.value, str):
            field_incoming_rule.value = (
                str(field_incoming_rule.value)
                if field_incoming_rule.value is not None
                else ""
            )
        elif type(field_incoming_rule.value) == int:
            field_incoming_rule.value = str(field_incoming_rule.value)
        elif isinstance(field_rule.value, list):
            try:
                parsed_value = json.loads(field_incoming_rule.value)
                if isinstance(parsed_value, list):
                    field_incoming_rule.value = parsed_value
            except (json.JSONDecodeError, ValueError):
                pass

    match = check_operator_match(field_rule, field_incoming_rule)
    return match


def find_best_match(
    objects: List[QueryObject], incoming_query: IncomingQuery
) -> Optional[QueryObject]:
    """
    Finds the best match for the incoming query among the list of objects.

    Args:
        objects (List[QueryObject]): The list of objects to search through.
        incomingQuery (IncomingQuery): The incoming query to match against.

    Returns:
        Optional[QueryObject]: The best match for the incoming query, or None if no match is found.
    """
    best_match = None
    max_match_count = 0
    incoming_query_rules = parse_incoming_query(incoming_query.query)
    for obj in objects:
        if evaluate_rule_group(obj.query, incoming_query_rules):
            if incoming_query.exactMatch and len(incoming_query_rules) != len(
                obj.query.rules
            ):
                continue
            match_count = len(obj.query.rules)
            if match_count > max_match_count:
                max_match_count = match_count
                best_match = obj
    return best_match


def find_all_matches(
    objects: List[QueryObject], incoming_query: IncomingQuery
) -> List[QueryObject]:
    """
    Finds all matches for the incoming query among the list of objects.

    Args:
        objects (List[QueryObject]): The list of objects to search through.
        incoming_query (IncomingQuery): The incoming query to match against.

    Returns:
        List[QueryObject]: A list of all matches for the incoming query.
    """
    matches: List[QueryObject] = []
    incoming_query_rules = parse_incoming_query(incoming_query.query)
    for obj in objects:
        if evaluate_rule_group(obj.query, incoming_query_rules):
            if incoming_query.exactMatch and len(incoming_query_rules) != len(
                obj.query.rules
            ):
                continue
            matches.append(obj)
    return matches
