from copy import copy
from itertools import combinations


def backtrack_sequences_with_cost(target: float,
                        costs: dict[str, float | int]) -> list[str]:
    """ 
    Creates all possible combinations of identifiers (keys of cost) so that the sum of the 
    costs gives the target.  
    """
    result: list[str] = []
    eps = 1e-9  # to handle float rounding

    def backtrack(current: str, total: float):
        if abs(total - target) < eps:
            result.append(current)
            return
        if total > target + eps:
            return

        for char, cost_value in costs.items():
            backtrack(current + char, total + cost_value)

    backtrack("", 0.0)
    return result

def find_unsupported_trajectories(costs: dict[str, int | float], partitioning: list[list[int |  float]]):
    """ 
    Find all the trajectories i.e. sequences of gates in the cost dict that are not supported by
    the provided partitioning.
    For Example: there are t and q gates in the costs dict, we built all the sequences of only t
    and q according to the total costs extracted from the partitioning.
    costs = {t:1.0, q:2.5}, from partitioning the total cost is 5
    sequences = [qq, ttttt]
    loop through the partitionings and check which of the above sequences is supported and which not    
    """
    targets = list(map(sum, partitioning))
    assert all(x == targets[0] for x in targets), "total cost must be equal for all input partitionings."
    
    sequences = backtrack_sequences_with_cost(targets[0], costs)
    
    unsupported = []
    for seq in sequences:
        budgets = [copy(part) for part in partitioning]
        if not any([supports_sequence(budget, costs, seq) for budget in budgets]):
            unsupported.append(seq)    
    
    return unsupported
            
def supports_sequence(budget: list[int | float], costs: dict[str, int | float], sequence: str):
    tol =  1e-8
    current_cost = 0.0
    current_index = 0
    current_cost_ceiling = budget[current_index]
    for char in sequence:
        if abs(current_cost_ceiling - current_cost) < tol:
            current_index += 1
            current_cost_ceiling = budget[current_index]
            current_cost = 0.0
            
        if (costs[char] - ( current_cost_ceiling - current_cost ) > tol 
                and abs(current_cost_ceiling - current_cost) > tol):
            # char doesn't fit in current bucket but bucket is not full yet
            # means sequence is not supported 
            return False
        
        # char fits in current bucket
        if costs[char] <= current_cost_ceiling - current_cost:
            current_cost += costs[char]
        
    return True


def pair_partitions(n: int | float, lower_bound: int| float, 
                    upper_bound: int| float, 
                    step: int | float):
    results = []
    # Start from ceil(x / step), end at floor(y / step)
    start = int((lower_bound + step - 1e-9) // step)  # Ceiling equivalent
    end = int(upper_bound // step)

    for i in range(start, end + 1):
        a = i * step
        b = n - a
        if abs(b - round(b / step) * step) < 1e-9:  # b is multiple of 0.5
            if lower_bound <= b <= upper_bound:
                results.append((a, b))
    return results


def min_valid_subset(original_set, f):
    elements = list(original_set)
    n = len(elements)

    best_subset = None
    best_size = 0

    def has_valid_subset(k):
        nonlocal best_subset, best_size
        if k == 0:
            if f(set()):
                best_subset = set()
                best_size = 0
            return f(set())
        
        for subset_tuple in combinations(elements, k):
            subset = set(subset_tuple)
            if f(subset):

                # Found valid! Update best
                if k < best_size or best_size == 0:
                    best_subset = subset
                    best_size = k
                return True  # Early exit: size k possible
        return False

    # Binary search for min size
    low, high = 0, n
    while low < high:
        mid = (low + high + 1) // 2
        if has_valid_subset(mid):
            high = mid - 1
        else:
            low = mid
            #high = mid - 1

    # Verify final result
    if best_size > 0:
        return best_size, best_subset

    return None, None # No valid subset exists


if __name__ == "__main__":
    # Example usage
    def f(s):
        return len(s) >= 2 and sum(s) > 10


    def is_valid_partition(partition):
        return find_unsupported_trajectories({'t': 1.0, "q": 2.5}, partition) == []


    unsupported = find_unsupported_trajectories({'t':1.0, "q": 2.5}, [[4.5, 5.0], [5.0, 4.5], [6, 3.5],[6.5, 3]])

    partitions = pair_partitions(9.5,1, 8, 0.5)
    print(pair_partitions(9.5,1, 8, 0.5))

    
    nums = {1, 2, 3, 4, 5, 6}
    size, subset = min_valid_subset(nums, f)
    print(f"Min size: {size}, Subset: {subset}")  # Min size: 2, Subset: {5, 6}
    
    _, min_subset =  min_valid_subset(partitions, is_valid_partition)
    print(list(map(list, min_subset)))

