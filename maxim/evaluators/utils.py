from ..models import (
    PassFailCriteria,
)


def sanitize_pass_fail_criteria(name: str, pass_fail_criteria: PassFailCriteria):
    all_operators = [">=", "<=", "<", ">", "=", "!="]
    boolean_operators = ["=", "!="]

    if isinstance(pass_fail_criteria.on_each_entry.value, (int, float)):
        if pass_fail_criteria.on_each_entry.score_should_be not in all_operators:
            raise Exception(
                f"Error While Creating Evaluator {name}: Invalid operator for scoreShouldBe, only accepts {', '.join(all_operators)}"
            )
    elif isinstance(pass_fail_criteria.on_each_entry.value, bool):
        if pass_fail_criteria.on_each_entry.score_should_be not in boolean_operators:
            raise Exception(
                f"Error While Creating Evaluator {name}: Invalid operator for scoreShouldBe, only accepts {', '.join(boolean_operators)}"
            )
    else:
        raise Exception(
            f"Error While Creating Evaluator {name}: Invalid type for onEachEntry.value, only accepts number or boolean"
        )

    if isinstance(pass_fail_criteria.for_testrun_overall.value, (int, float)):
        if (
            pass_fail_criteria.for_testrun_overall.overall_should_be
            not in all_operators
        ):
            raise Exception(
                f"Error While Creating Evaluator {name}: Invalid operator for overallShouldBe, only accepts {', '.join(all_operators)}"
            )
        if (
            pass_fail_criteria.for_testrun_overall.for_result != "average"
            and pass_fail_criteria.for_testrun_overall.for_result
            != "percentageOfPassedResults"
        ):
            raise Exception(
                f"Error While Creating Evaluator {name}: Invalid value for `for` in forTestrunOverall, only accepts 'average' or 'percentageOfPassedResults'"
            )
    else:
        raise Exception(
            f"Error While Creating Evaluator {name}: Invalid type for forTestrunOverall.value, only accepts number"
        )
