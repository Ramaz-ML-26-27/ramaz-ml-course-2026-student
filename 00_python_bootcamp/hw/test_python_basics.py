"""Tests for HW00 Part 1 — python_basics.py

Each class tests one function. All tests in a class must pass to earn credit.
Run with: uv run pytest
"""

from __future__ import annotations

import pytest

from python_basics import (
    Gradebook,
    Student,
    apply_twice,
    chunk,
    count_occurrences,
    deep_get,
    find_duplicates,
    flatten,
    group_by,
    invert_dict,
    jaccard_similarity,
    make_multiplier,
    memoize,
    most_frequent,
    pipeline,
    rotate,
    run_length_encode,
    running_average,
    sliding_window,
    two_sum,
)

# ── Part 1: Lists ─────────────────────────────────────────────────────────────


@pytest.mark.lists
class TestFlatten:
    def test_basic(self) -> None:
        result = flatten([[1, 2], [3, 4], [5]])
        assert result == [1, 2, 3, 4, 5], (
            f"flatten([[1, 2], [3, 4], [5]]) should return [1, 2, 3, 4, 5]; got {result}"
        )

    def test_empty_sublist(self) -> None:
        result = flatten([[], [1]])
        assert result == [1], f"Empty sublists should be skipped; got {result}"

    def test_empty_outer(self) -> None:
        result = flatten([])
        assert result == [], f"flatten([]) should return []; got {result}"

    def test_single_sublist(self) -> None:
        result = flatten([[1, 2, 3]])
        assert result == [1, 2, 3], f"Single sublist should be unwrapped; got {result}"

    def test_strings(self) -> None:
        result = flatten([["a", "b"], ["c"]])
        assert result == ["a", "b", "c"], f"Should work with strings; got {result}"


@pytest.mark.lists
class TestMostFrequent:
    def test_integers(self) -> None:
        result = most_frequent([1, 2, 2, 3])
        assert result == 2, f"Expected 2 (appears twice); got {result}"

    def test_strings(self) -> None:
        result = most_frequent(["a", "b", "a", "c", "a"])
        assert result == "a", f"Expected 'a' (appears 3 times); got {result}"

    def test_single_element(self) -> None:
        result = most_frequent([42])
        assert result == 42, f"Single-element list: expected 42; got {result}"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            most_frequent([])


@pytest.mark.lists
class TestRunningAverage:
    def test_basic(self) -> None:
        result = running_average([10.0, 20.0, 30.0])
        assert result == [10.0, 15.0, 20.0], (
            f"Expected [10.0, 15.0, 20.0]; got {result}"
        )

    def test_single(self) -> None:
        result = running_average([4.0])
        assert result == [4.0], f"Single element: expected [4.0]; got {result}"

    def test_empty(self) -> None:
        result = running_average([])
        assert result == [], f"Empty list: expected []; got {result}"

    def test_four_elements(self) -> None:
        result = running_average([1.0, 2.0, 3.0, 4.0])
        assert result == [1.0, 1.5, 2.0, 2.5], (
            f"Expected [1.0, 1.5, 2.0, 2.5]; got {result}"
        )

    def test_length_matches_input(self) -> None:
        nums = [5.0, 3.0, 8.0, 2.0, 7.0]
        result = running_average(nums)
        assert len(result) == len(nums), (
            f"Output length {len(result)} should match input length {len(nums)}"
        )


@pytest.mark.lists
class TestChunk:
    def test_basic(self) -> None:
        result = chunk([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]], (
            f"Expected [[1, 2], [3, 4], [5]]; got {result}"
        )

    def test_exact_fit(self) -> None:
        result = chunk([1, 2, 3], 3)
        assert result == [[1, 2, 3]], f"Expected [[1, 2, 3]]; got {result}"

    def test_empty(self) -> None:
        result = chunk([], 4)
        assert result == [], f"Empty list: expected []; got {result}"

    def test_size_one(self) -> None:
        result = chunk([1, 2, 3], 1)
        assert result == [[1], [2], [3]], (
            f"Size-1 chunks: expected [[1], [2], [3]]; got {result}"
        )

    def test_even_split(self) -> None:
        result = chunk([1, 2, 3, 4], 2)
        assert result == [[1, 2], [3, 4]], (
            f"Even split: expected [[1, 2], [3, 4]]; got {result}"
        )


@pytest.mark.lists
class TestRotate:
    def test_left_basic(self) -> None:
        result = rotate([1, 2, 3, 4, 5], 2)
        assert result == [3, 4, 5, 1, 2], (
            f"Rotate left by 2: expected [3, 4, 5, 1, 2]; got {result}"
        )

    def test_right(self) -> None:
        result = rotate([1, 2, 3], -1)
        assert result == [3, 1, 2], (
            f"Rotate right by 1 (k=-1): expected [3, 1, 2]; got {result}"
        )

    def test_wraps_around(self) -> None:
        result = rotate([1, 2, 3], 4)
        assert result == [2, 3, 1], (
            f"k=4 wraps (same as k=1): expected [2, 3, 1]; got {result}"
        )

    def test_empty(self) -> None:
        result = rotate([], 3)
        assert result == [], f"Empty list: expected []; got {result}"

    def test_zero(self) -> None:
        result = rotate([1, 2, 3], 0)
        assert result == [1, 2, 3], f"k=0: list should be unchanged; got {result}"

    def test_full_rotation(self) -> None:
        result = rotate([1, 2, 3], 3)
        assert result == [1, 2, 3], (
            f"k=len(items): list should be unchanged; got {result}"
        )


@pytest.mark.lists
class TestRunLengthEncode:
    def test_basic(self) -> None:
        result = run_length_encode(["a", "a", "b", "b", "b", "a"])
        assert result == [(2, "a"), (3, "b"), (1, "a")], (
            f"Expected [(2, 'a'), (3, 'b'), (1, 'a')]; got {result}"
        )

    def test_no_consecutive(self) -> None:
        result = run_length_encode([1, 2, 3])
        assert result == [(1, 1), (1, 2), (1, 3)], (
            f"All unique: expected [(1, 1), (1, 2), (1, 3)]; got {result}"
        )

    def test_empty(self) -> None:
        result = run_length_encode([])
        assert result == [], f"Empty list: expected []; got {result}"

    def test_single(self) -> None:
        result = run_length_encode(["x"])
        assert result == [(1, "x")], (
            f"Single element: expected [(1, 'x')]; got {result}"
        )

    def test_all_same(self) -> None:
        result = run_length_encode([1, 1, 1, 1])
        assert result == [(4, 1)], f"All same: expected [(4, 1)]; got {result}"

    def test_alternating(self) -> None:
        result = run_length_encode(["x", "y", "x", "y"])
        assert result == [(1, "x"), (1, "y"), (1, "x"), (1, "y")], (
            f"Alternating: each run has length 1; got {result}"
        )


@pytest.mark.lists
class TestSlidingWindow:
    def test_basic(self) -> None:
        result = sliding_window([1, 2, 3, 4, 5], 3)
        assert result == [[1, 2, 3], [2, 3, 4], [3, 4, 5]], (
            f"Expected [[1,2,3],[2,3,4],[3,4,5]]; got {result}"
        )

    def test_size_two(self) -> None:
        result = sliding_window([1, 2, 3], 2)
        assert result == [[1, 2], [2, 3]], (
            f"Expected [[1,2],[2,3]]; got {result}"
        )

    def test_size_larger_than_list(self) -> None:
        result = sliding_window([1, 2], 3)
        assert result == [], (
            f"size > len(items): expected []; got {result}"
        )

    def test_empty(self) -> None:
        result = sliding_window([], 2)
        assert result == [], f"Empty list: expected []; got {result}"

    def test_exact_fit(self) -> None:
        result = sliding_window([1, 2, 3], 3)
        assert result == [[1, 2, 3]], (
            f"Window fills entire list: expected [[1,2,3]]; got {result}"
        )

    def test_window_count(self) -> None:
        result = sliding_window(list(range(10)), 4)
        assert len(result) == 7, (
            f"10 items, window size 4 → 7 windows; got {len(result)}"
        )


# ── Part 2: Dicts ──────────────────────────────────────────────────────────────


@pytest.mark.dicts
class TestCountOccurrences:
    def test_basic(self) -> None:
        result = count_occurrences(["a", "b", "a", "c", "b", "b"])
        assert result == {"a": 2, "b": 3, "c": 1}, (
            f"Expected {{'a': 2, 'b': 3, 'c': 1}}; got {result}"
        )

    def test_empty(self) -> None:
        result = count_occurrences([])
        assert result == {}, f"Empty list: expected {{}}; got {result}"

    def test_all_same(self) -> None:
        result = count_occurrences([1, 1, 1])
        assert result == {1: 3}, f"All same: expected {{1: 3}}; got {result}"

    def test_specific_counts(self) -> None:
        result = count_occurrences([3, 1, 4, 1, 5, 9, 2, 6, 5])
        assert result.get(1) == 2, f"1 appears twice; got count {result.get(1)}"
        assert result.get(5) == 2, f"5 appears twice; got count {result.get(5)}"
        assert result.get(3) == 1, f"3 appears once; got count {result.get(3)}"


@pytest.mark.dicts
class TestInvertDict:
    def test_basic(self) -> None:
        result = invert_dict({"a": 1, "b": 2})
        assert result == {1: "a", 2: "b"}, f"Expected {{1: 'a', 2: 'b'}}; got {result}"

    def test_empty(self) -> None:
        result = invert_dict({})
        assert result == {}, f"Empty dict: expected {{}}; got {result}"

    def test_keys_and_values_swapped(self) -> None:
        original = {"x": 10, "y": 20, "z": 30}
        result = invert_dict(original)
        assert set(result.keys()) == {10, 20, 30}, (
            f"Keys of inverted dict should be the original values; got {set(result.keys())}"
        )
        assert set(result.values()) == {"x", "y", "z"}, (
            f"Values of inverted dict should be the original keys; got {set(result.values())}"
        )


@pytest.mark.dicts
class TestGroupBy:
    def test_basic(self) -> None:
        records = [
            {"name": "Alice", "dept": "Eng"},
            {"name": "Bob", "dept": "HR"},
            {"name": "Carol", "dept": "Eng"},
        ]
        result = group_by(records, "dept")
        assert set(result.keys()) == {"Eng", "HR"}, (
            f"Expected groups 'Eng' and 'HR'; got {set(result.keys())}"
        )
        assert len(result["Eng"]) == 2, (
            f"'Eng' group should have 2 records; got {len(result['Eng'])}"
        )
        assert len(result["HR"]) == 1, (
            f"'HR' group should have 1 record; got {len(result['HR'])}"
        )

    def test_different_key(self) -> None:
        records = [
            {"genre": "Pop", "title": "Song A"},
            {"genre": "Rock", "title": "Song B"},
            {"genre": "Pop", "title": "Song C"},
        ]
        result = group_by(records, "genre")
        assert len(result["Pop"]) == 2, "Pop should have 2 songs"
        assert len(result["Rock"]) == 1, "Rock should have 1 song"

    def test_all_same_group(self) -> None:
        records = [{"cat": "A", "val": i} for i in range(4)]
        result = group_by(records, "cat")
        assert len(result) == 1, (
            f"All records share one group; got {len(result)} groups"
        )
        assert len(result["A"]) == 4, (
            f"All 4 records in group 'A'; got {len(result['A'])}"
        )

    def test_preserves_full_record(self) -> None:
        records = [{"name": "Alice", "dept": "Eng", "score": 90}]
        result = group_by(records, "dept")
        assert result["Eng"][0] == {"name": "Alice", "dept": "Eng", "score": 90}, (
            "Full record should be preserved in the group (no fields dropped)"
        )


@pytest.mark.dicts
class TestDeepGet:
    def test_basic(self) -> None:
        result = deep_get({"a": {"b": {"c": 42}}}, "a.b.c")
        assert result == 42, f"Expected 42; got {result}"

    def test_missing_key_returns_default(self) -> None:
        result = deep_get({"a": 1}, "a.b", default=-1)
        assert result == -1, f"Missing key should return default -1; got {result}"

    def test_missing_top_returns_none(self) -> None:
        result = deep_get({"x": 10}, "y")
        assert result is None, (
            f"Missing top-level key should return None (the default); got {result}"
        )

    def test_single_key(self) -> None:
        result = deep_get({"a": 99}, "a")
        assert result == 99, f"Single-key path: expected 99; got {result}"

    def test_zero_value_not_confused_with_default(self) -> None:
        result = deep_get({"a": {"b": 0}}, "a.b", default=-1)
        assert result == 0, (
            f"A stored value of 0 should not be confused with the default; got {result}"
        )

    def test_non_dict_along_path(self) -> None:
        result = deep_get({"a": 5}, "a.b", default="missing")
        assert result == "missing", (
            f"Reaching a non-dict value mid-path should return default; got {result}"
        )


@pytest.mark.dicts
class TestTwoSum:
    def test_basic(self) -> None:
        result = two_sum([2, 7, 11, 15], 9)
        assert result == (0, 1), f"Expected (0, 1); got {result}"

    def test_not_first_pair(self) -> None:
        result = two_sum([3, 2, 4], 6)
        assert result == (1, 2), f"Expected (1, 2); got {result}"

    def test_no_solution(self) -> None:
        result = two_sum([1, 2, 3], 100)
        assert result is None, f"No solution: expected None; got {result}"

    def test_index_ordering(self) -> None:
        result = two_sum([5, 1, 4], 5)
        assert result is not None, "Should find 1+4=5 at indices 1 and 2"
        i, j = result
        assert i < j, (
            f"First index must be less than second index (i < j); got ({i}, {j})"
        )

    def test_negative_numbers(self) -> None:
        nums = [-1, 0, 1, 2]
        result = two_sum(nums, 1)
        assert result is not None, "Should find a pair summing to 1 (-1 + 2 = 1)"
        i, j = result
        assert nums[i] + nums[j] == 1, (
            f"nums[{i}] + nums[{j}] should equal 1; got {nums[i] + nums[j]}"
        )


# ── Part 3: Sets ───────────────────────────────────────────────────────────────


@pytest.mark.sets
class TestFindDuplicates:
    def test_basic(self) -> None:
        result = find_duplicates([1, 2, 2, 3, 3, 3, 4])
        assert result == {2, 3}, f"Expected {{2, 3}}; got {result}"

    def test_no_duplicates(self) -> None:
        result = find_duplicates([1, 2, 3])
        assert result == set(), f"No duplicates: expected empty set; got {result}"

    def test_empty(self) -> None:
        result = find_duplicates([])
        assert result == set(), f"Empty list: expected empty set; got {result}"

    def test_all_duplicates(self) -> None:
        result = find_duplicates([1, 1, 2, 2, 3, 3])
        assert result == {1, 2, 3}, f"Expected {{1, 2, 3}}; got {result}"

    def test_returns_set_type(self) -> None:
        result = find_duplicates([1, 1, 2])
        assert isinstance(result, set), (
            f"Return type should be set, not {type(result).__name__}"
        )


@pytest.mark.sets
class TestJaccardSimilarity:
    def test_partial_overlap(self) -> None:
        result = jaccard_similarity({1, 2, 3}, {2, 3, 4})
        assert abs(result - 0.5) < 1e-9, f"|A∩B|=2, |A∪B|=4, expected 0.5; got {result}"

    def test_no_overlap(self) -> None:
        result = jaccard_similarity({1, 2}, {3, 4})
        assert result == 0.0, f"Disjoint sets: expected 0.0; got {result}"

    def test_both_empty(self) -> None:
        result = jaccard_similarity(set(), set())
        assert result == 0.0, f"Both empty: expected 0.0; got {result}"

    def test_identical(self) -> None:
        result = jaccard_similarity({1, 2, 3}, {1, 2, 3})
        assert abs(result - 1.0) < 1e-9, f"Identical sets: expected 1.0; got {result}"

    def test_one_empty(self) -> None:
        result = jaccard_similarity({1, 2}, set())
        assert result == 0.0, f"One empty set: expected 0.0; got {result}"

    def test_subset(self) -> None:
        result = jaccard_similarity({1, 2}, {1, 2, 3, 4})
        # intersection={1,2}, union={1,2,3,4}, similarity=2/4=0.5
        assert abs(result - 0.5) < 1e-9, (
            f"Subset: intersection=2, union=4, expected 0.5; got {result}"
        )


# ── Part 4: Higher-order functions ────────────────────────────────────────────


@pytest.mark.hof
class TestApplyTwice:
    def test_double(self) -> None:
        result = apply_twice(lambda n: n * 2, 3)
        assert result == 12, (
            f"apply_twice(double, 3) = double(double(3)) = 12; got {result}"
        )

    def test_upper(self) -> None:
        result = apply_twice(str.upper, "hello")
        assert result == "HELLO", (
            f"apply_twice(str.upper, 'hello'): 'hello' is already uppercase after first call; "
            f"got {result}"
        )

    def test_add_one(self) -> None:
        result = apply_twice(lambda x: x + 1, 10)
        assert result == 12, f"apply_twice(add1, 10) = 12; got {result}"


@pytest.mark.hof
class TestMakeMultiplier:
    def test_double(self) -> None:
        double = make_multiplier(2)
        result = double(5)
        assert result == 10.0, f"double(5) should be 10.0; got {result}"

    def test_triple(self) -> None:
        triple = make_multiplier(3)
        result = triple(4)
        assert result == 12.0, f"triple(4) should be 12.0; got {result}"

    def test_independence(self) -> None:
        double = make_multiplier(2)
        triple = make_multiplier(3)
        assert double(5) == 10.0, "double should be 10.0 regardless of triple"
        assert triple(5) == 15.0, "triple should be 15.0 (closures are independent)"

    def test_fractional(self) -> None:
        half = make_multiplier(0.5)
        result = half(8)
        assert result == 4.0, f"half(8) should be 4.0; got {result}"


@pytest.mark.hof
class TestPipeline:
    def test_two_functions(self) -> None:
        def add1(x: int) -> int:
            return x + 1

        def double(x: int) -> int:
            return x * 2

        result = pipeline(add1, double)(3)
        assert result == 8, (
            f"pipeline(add1, double)(3) = double(add1(3)) = double(4) = 8; got {result}"
        )

    def test_identity_when_empty(self) -> None:
        result = pipeline()(42)
        assert result == 42, (
            f"pipeline() with no functions should be the identity; got {result}"
        )

    def test_three_functions(self) -> None:
        def add1(x: int) -> int:
            return x + 1

        def double(x: int) -> int:
            return x * 2

        def negate(x: int) -> int:
            return -x

        result = pipeline(add1, double, negate)(3)
        # add1(3)=4, double(4)=8, negate(8)=-8
        assert result == -8, (
            f"pipeline(add1, double, negate)(3): 3->4->8->-8; got {result}"
        )

    def test_single_function(self) -> None:
        def square(x: int) -> int:
            return x**2

        result = pipeline(square)(5)
        assert result == 25, f"pipeline(square)(5) = 25; got {result}"

    def test_order_matters(self) -> None:
        def add1(x: int) -> int:
            return x + 1

        def double(x: int) -> int:
            return x * 2

        result_ab = pipeline(add1, double)(3)  # (3+1)*2 = 8
        result_ba = pipeline(double, add1)(3)  # 3*2+1 = 7
        assert result_ab == 8, f"pipeline(add1, double)(3) should be 8; got {result_ab}"
        assert result_ba == 7, f"pipeline(double, add1)(3) should be 7; got {result_ba}"


@pytest.mark.hof
class TestMemoize:
    def test_returns_correct_result(self) -> None:
        def square(x: int) -> int:
            return x**2

        cached_square = memoize(square)
        assert cached_square(4) == 16, (
            f"memoize should return the correct result; got {cached_square(4)}"
        )

    def test_does_not_call_function_twice(self) -> None:
        call_count = 0

        def tracked(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x**2

        cached = memoize(tracked)
        cached(4)
        cached(4)  # should NOT call tracked again
        assert call_count == 1, (
            f"Function should only be called once for repeated args; "
            f"was called {call_count} times"
        )

    def test_different_args_call_again(self) -> None:
        call_count = 0

        def tracked(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        cached = memoize(tracked)
        cached(3)
        cached(5)  # different arg — should call function again
        assert call_count == 2, (
            f"Different arguments should not share cached results; "
            f"expected 2 calls, got {call_count}"
        )

    def test_many_repeated_calls(self) -> None:
        call_count = 0

        def tracked(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + 1

        cached = memoize(tracked)
        for _ in range(10):
            cached(7)
        assert call_count == 1, (
            f"Calling with the same arg 10 times should only invoke the function once; "
            f"got {call_count} calls"
        )


# ── Part 5: Classes ───────────────────────────────────────────────────────────


@pytest.mark.classes
class TestStudent:
    def test_average_basic(self) -> None:
        s = Student("Alice", [80.0, 90.0, 70.0])
        assert s.average() == 80.0, (
            f"Average of [80, 90, 70] should be 80.0; got {s.average()}"
        )

    def test_average_empty(self) -> None:
        s = Student("Bob", [])
        assert s.average() == 0.0, (
            f"Empty grades: average should be 0.0; got {s.average()}"
        )

    def test_highest_basic(self) -> None:
        s = Student("Carol", [70.0, 95.0, 85.0])
        assert s.highest() == 95.0, (
            f"Highest of [70, 95, 85] should be 95.0; got {s.highest()}"
        )

    def test_highest_empty(self) -> None:
        s = Student("Dave", [])
        assert s.highest() == 0.0, (
            f"Empty grades: highest should be 0.0; got {s.highest()}"
        )

    def test_letter_grade_a(self) -> None:
        s = Student("Eve", [95.0, 92.0])
        assert s.letter_grade() == "A", (
            f"Average 93.5: expected 'A'; got {s.letter_grade()}"
        )

    def test_letter_grade_b(self) -> None:
        s = Student("Frank", [80.0, 85.0])
        assert s.letter_grade() == "B", (
            f"Average 82.5: expected 'B'; got {s.letter_grade()}"
        )

    def test_letter_grade_c(self) -> None:
        s = Student("Grace", [70.0, 75.0])
        assert s.letter_grade() == "C", (
            f"Average 72.5: expected 'C'; got {s.letter_grade()}"
        )

    def test_letter_grade_d(self) -> None:
        s = Student("Hank", [60.0, 65.0])
        assert s.letter_grade() == "D", (
            f"Average 62.5: expected 'D'; got {s.letter_grade()}"
        )

    def test_letter_grade_f(self) -> None:
        s = Student("Iris", [30.0, 50.0])
        assert s.letter_grade() == "F", (
            f"Average 40.0: expected 'F'; got {s.letter_grade()}"
        )

    def test_repr_contains_name_and_avg(self) -> None:
        s = Student("Alice", [80.0, 90.0])
        r = repr(s)
        assert "Alice" in r, f"__repr__ should include the name; got {r!r}"
        assert "85.0" in r, f"__repr__ should include the average (85.0); got {r!r}"

    def test_comparison_lt(self) -> None:
        low = Student("Low", [60.0])
        high = Student("High", [90.0])
        assert low < high, (
            "Student with lower average should be less than student with higher average"
        )
        assert not (high < low), (
            "Student with higher average should not be less than student with lower average"
        )

    def test_sorted(self) -> None:
        students = [
            Student("C", [75.0]),
            Student("A", [90.0]),
            Student("B", [80.0]),
        ]
        result = sorted(students)
        assert [s.name for s in result] == ["C", "B", "A"], (
            f"sorted() should order by average ascending; "
            f"got names in order {[s.name for s in result]}"
        )


@pytest.mark.classes
class TestGradebook:
    def test_add_student(self) -> None:
        gb = Gradebook()
        s = Student("Alice", [85.0, 90.0])
        gb.add_student(s)
        assert "Alice" in gb.students, (
            "Added student should appear in gradebook.students"
        )

    def test_duplicate_raises_value_error(self) -> None:
        gb = Gradebook()
        gb.add_student(Student("Alice", [80.0]))
        with pytest.raises(ValueError):
            gb.add_student(Student("Alice", [90.0]))

    def test_top_students_order(self) -> None:
        gb = Gradebook()
        gb.add_student(Student("Low", [60.0]))
        gb.add_student(Student("High", [95.0]))
        gb.add_student(Student("Mid", [78.0]))
        top = gb.top_students(2)
        assert top[0].name == "High", f"Top student should be 'High'; got {top[0].name}"
        assert top[1].name == "Mid", (
            f"Second student should be 'Mid'; got {top[1].name}"
        )

    def test_top_students_count(self) -> None:
        gb = Gradebook()
        for i in range(5):
            gb.add_student(Student(f"S{i}", [float(i * 10 + 50)]))
        top3 = gb.top_students(3)
        assert len(top3) == 3, (
            f"top_students(3) should return 3 students; got {len(top3)}"
        )

    def test_class_average_basic(self) -> None:
        gb = Gradebook()
        gb.add_student(Student("A", [80.0]))
        gb.add_student(Student("B", [90.0]))
        result = gb.class_average()
        assert abs(result - 85.0) < 1e-9, (
            f"Average of students averaging 80 and 90 should be 85.0; got {result}"
        )

    def test_class_average_empty(self) -> None:
        gb = Gradebook()
        result = gb.class_average()
        assert result == 0.0, (
            f"Empty gradebook: class_average should be 0.0; got {result}"
        )
