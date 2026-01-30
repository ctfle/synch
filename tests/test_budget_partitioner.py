import pytest
from trasyn.synthesis import BudgetPartitioner


class TestBudgetPartitioner:
    def test_non_integer_budgets(self):
        costs = {"T": 1.0, "sqrtT": 2.5}
        budget = BudgetPartitioner(
            costs=costs, total_non_clifford_budget=10, max_partition_value=5
        )
        partition = budget.partition()
        assert partition == [
            [0.0],
            [1.0],
            [2.0],
            [2.5],
            [3.0],
            [3.5],
            [4.0],
            [4.5],
            [5.0],
            [2.5, 3.0],
            [5.0, 1.0],
            [2.5, 4.0],
            [5.0, 2.0],
            [2.5, 5.0],
            [5.0, 3.0],
            [2.5, 5.0, 1.0],
            [5.0, 4.0],
            [2.5, 5.0, 2.0],
            [5.0, 5.0],
        ]

    def test_integer_budgets_3(self):
        costs = {"T": 1.0, "sqrtT": 3.0}
        budget = BudgetPartitioner(
            costs=costs, total_non_clifford_budget=10, max_partition_value=6
        )
        partition = budget.partition()
        assert partition == [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [6.0, 1.0],
            [6.0, 2.0],
            [6.0, 3.0],
            [6.0, 4.0],
        ]

    def test_integer_budgets_2(self):
        costs = {"T": 1.0, "sqrtT": 2.0}
        budget = BudgetPartitioner(
            costs=costs, total_non_clifford_budget=10, max_partition_value=4
        )
        partition = budget.partition()
        assert partition == [
            [0],
            [1],
            [2],
            [3],
            [4],
            [4.0, 1.0],
            [4.0, 2.0],
            [4.0, 3.0],
            [4.0, 4.0],
            [4.0, 4.0, 1.0],
            [4.0, 4.0, 2.0],
        ]

    def test_integets_2(self):
        costs = {"T": 1.0, "sqrtT": 3.0}
        budget = BudgetPartitioner(
            costs=costs, total_non_clifford_budget=6, max_partition_value=4
        )
        partition = budget.partition()
        print(partition)

    @pytest.mark.parametrize("max_partition_value", [1, 2, 3, 4])
    def test_max_partition_value_too_small(self, max_partition_value: int):
        costs = {"T": 1.0, "sqrtT": 2.5}
        with pytest.raises(AssertionError):
            budget = BudgetPartitioner(
                costs=costs,
                total_non_clifford_budget=10,
                max_partition_value=max_partition_value,
            )

    @pytest.mark.parametrize("cost", [1.1, 2.2, 3.3, 4.4])
    def test_wrong_cost_values(self, cost: int):
        costs = {"some_gate1": 1.0, "some_gate2": cost}
        with pytest.raises(AssertionError):
            budget = BudgetPartitioner(
                costs=costs, total_non_clifford_budget=10, max_partition_value=10
            )
