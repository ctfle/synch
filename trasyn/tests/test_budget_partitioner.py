import pytest
from trasyn.synthesis import ErgodicPartitioner


class TestErgodicPartitioner:
    def test_non_integer_budgets(self):
        costs = {"t": 1.0, "q": 2.5}
        budget = ErgodicPartitioner(
            costs=costs, total_non_clifford_budget=10, max_partition_value=8
        )
        partition = budget.partition()
        
        expected = [
            [0.0],
            [1.0],
            [2.0],
            [2.5],
            [3.0],
            [3.5],
            [4.0],
            [4.5],
            [5.0],
            [5.5],
            [6.0],
            [6.5],
            [7.0],
            [7.5],
            [1.0, 7.0],
            [2.5, 5.5],
            [1.0, 7.5],
            [2.5, 6.0],
            [1.0, 8.0],
            [2.5, 6.5],
            [2.5, 7.0],
            [3.5, 6.0],
            [2.0, 7.5],
            [2.5, 7.5],
            [3.5, 6.5],
            [2.0, 8.0],
        ]
        expected = list(map(set, expected))
        partition = list(map(set, partition))
        for partition_element in partition:
            assert partition_element in expected
    

    def test_integer_budgets_3(self):
        costs = {"t": 1.0, "a": 3.0}
        budget = ErgodicPartitioner(
            costs=costs, total_non_clifford_budget=10, max_partition_value=6
        )
        partition = budget.partition()
        expected = [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [3, 3],
            [1, 5],
            [1, 6],
            [3, 4],
            [4, 4],
            [2, 6],
            [3, 5],
            [4, 5],
            [5, 4],
            [3, 6],
            [4, 6],
            [6, 4],
            [5, 5],
        ]
    
        expected = list(map(set, expected))
        partition = list(map(set, partition))
        for partition_element in partition:
            assert partition_element in expected

    def test_integer_budgets_2(self):
        costs = {"T": 1.0, "q": 2.0}
        budget = ErgodicPartitioner(
            costs=costs, total_non_clifford_budget=10, max_partition_value=8
        )
        partition = budget.partition()
        expected = [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [1, 7],
            [2, 6],
            [1, 8],
            [2, 7],
            [3, 7],
            [2, 8],
        ]
        
        expected = list(map(set, expected))
        partition = list(map(set, partition))
        for partition_element in partition:
            assert partition_element in expected

    def test_integer_budgets_without_permutations(self):
        costs = {"t": 1.0, "q": 2.0}
        budget = ErgodicPartitioner(
            costs=costs,
            total_non_clifford_budget=10,
            max_partition_value=8,
        )
        partition = budget.partition()
        expected = [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [1, 7],
            [2, 6],
            [1, 8],
            [2, 7],
            [3, 7],
            [2, 8],
        ]
        
        expected = list(map(set, expected))
        partition = list(map(set, partition))
        for partition_element in partition:
            assert partition_element in expected

    @pytest.mark.parametrize("max_partition_value", [1, 2, 3, 4])
    def test_max_partition_value_too_small(self, max_partition_value: int):
        costs = {"T": 1.0, "sqrtT": 2.5}
        with pytest.raises(AssertionError):
            budget = ErgodicPartitioner(
                costs=costs,
                total_non_clifford_budget=10,
                max_partition_value=max_partition_value,
            )

    @pytest.mark.parametrize("cost", [1.1, 2.2, 3.3, 4.4])
    def test_wrong_cost_values(self, cost: int):
        costs = {"some_gate1": 1.0, "some_gate2": cost}
        with pytest.raises(AssertionError):
            budget = ErgodicPartitioner(
                costs=costs, total_non_clifford_budget=10, max_partition_value=10
            )
