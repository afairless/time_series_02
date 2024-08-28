 
# import pytest
import numpy as np

from src.s01_generate_data import (
    randomize_segment_lengths,
    flatten_list_of_lists,
    generate_and_combine_trends,
    )


def test_randomize_segment_lengths_01():
    """
    Test valid input
    """
    seed = 211612
    total_n = 10
    segment_n = 2
    result = randomize_segment_lengths(total_n, segment_n, seed)
    assert len(result) == segment_n
    assert result.sum() == total_n 


def test_randomize_segment_lengths_02():
    """
    Test several combinations of valid input
    """

    seed = 211612
    total_n = range(3, 400, 13)
    segment_n = range(1, 15, 3)

    for total in total_n:
        for segment in segment_n:
            if segment < total:
                result = randomize_segment_lengths(total, segment, seed)
                assert len(result) == segment
                assert result.sum() == total


def test_flatten_list_of_lists_01():
    """
    Test valid input
    """
    a_list = [[], [], []]
    result = flatten_list_of_lists(a_list)
    correct_result = []
    assert result == correct_result


def test_flatten_list_of_lists_02():
    """
    Test valid input
    """
    a_list = [[1, 2], [3, 4], [5, 6]]
    result = flatten_list_of_lists(a_list)
    correct_result = [1, 2, 3, 4, 5, 6]
    assert result == correct_result


def test_flatten_list_of_lists_03():
    """
    Test valid input
    """
    a_list = [[1], [2, 3, 4], [], [5, 6]]
    result = flatten_list_of_lists(a_list)
    correct_result = [1, 2, 3, 4, 5, 6]
    assert result == correct_result


def test_generate_and_combine_trends_01():
    """
    Test valid input
    """

    seed = 827816
    time_n = 10
    trend_n = 1
    trend_slope_min = -1 
    trend_slope_max = 1

    result_all = generate_and_combine_trends(
        time_n, trend_n, trend_slope_min, trend_slope_max, seed)
    result = result_all.combined_trend

    assert len(result) == time_n

    step_diff = result[1:] - result[:-1]
    np.testing.assert_almost_equal(np.std(step_diff), 0)

    assert trend_slope_min <= step_diff[0] <= trend_slope_max


def test_generate_and_combine_trends_02():
    """
    Test valid input
    """

    seed = 789812
    time_n = 15
    trend_n = 2
    trend_slope_min = 3 
    trend_slope_max = 3.03

    result_all = generate_and_combine_trends(
        time_n, trend_n, trend_slope_min, trend_slope_max, seed)
    result = result_all.combined_trend

    assert len(result) == time_n

    step_diff = np.unique(result[1:] - result[:-1])

    for diff in step_diff:
        assert trend_slope_min <= diff <= trend_slope_max

    step_diff_diff = step_diff[1:] - step_diff[:-1]
    unique_values_minus_one = (step_diff_diff > 1e-6).sum()
    np.testing.assert_equal(unique_values_minus_one, trend_n - 1)


def test_generate_and_combine_trends_03():
    """
    Test valid input
    """

    seed = 789812
    time_n = 937
    trend_n = 13
    trend_slope_min = -8 
    trend_slope_max = -7

    result_all = generate_and_combine_trends(
        time_n, trend_n, trend_slope_min, trend_slope_max, seed)
    result = result_all.combined_trend

    assert len(result) == time_n

    step_diff = np.unique(result[1:] - result[:-1])

    for diff in step_diff:
        assert trend_slope_min <= diff <= trend_slope_max

    step_diff_diff = step_diff[1:] - step_diff[:-1]
    unique_values_minus_one = (step_diff_diff > 1e-6).sum()
    np.testing.assert_equal(unique_values_minus_one, trend_n - 1)


# def test_dummy_function02_01():
#     """
#     Test input of wrong data type
#     """

#     input = -1

#     with pytest.raises(AssertionError):
#         result = dummy_function02(input)


