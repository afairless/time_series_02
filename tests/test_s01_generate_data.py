 
import pytest
import numpy as np

from src.s01_generate_data import (
    randomize_segment_lengths,
    flatten_list_of_lists,
    expand_values_by_lengths_into_vector,
    generate_and_combine_trends,
    )


def test_randomize_segment_lengths_01():
    """
    Test valid input with default value for 'last_segment_len'
    """
    seed = 211612
    total_n = 10
    segment_n = 2
    result = randomize_segment_lengths(total_n, segment_n, seed=seed)
    assert len(result) == segment_n
    assert result.sum() == total_n 


def test_randomize_segment_lengths_02():
    """
    Test several combinations of valid input with default value for 
        'last_segment_len'
    """

    seed = 211612
    total_n = range(3, 400, 13)
    segment_n = range(1, 15, 3)

    for total in total_n:
        for segment in segment_n:
            if segment < total:
                result = randomize_segment_lengths(total, segment, seed=seed)
                assert len(result) == segment
                assert result.sum() == total


def test_randomize_segment_lengths_03():
    """
    Test valid input with provided value for 'last_segment_len'
    """
    seed = 193945
    total_n = 1000
    segment_n = 7
    last_segment_len = 197
    result = randomize_segment_lengths(
        total_n, segment_n, last_segment_len, seed)
    assert len(result) == segment_n
    assert result.sum() == total_n 


def test_randomize_segment_lengths_04():
    """
    Test several combinations of valid input with provided value for 
        'last_segment_len'
    """

    seed = 315912
    total_n = range(20, 600, 13)
    segment_n = range(1, 15, 3)
    last_segment_len = range(10, 300, 7)

    for total in total_n:
        for segment in segment_n:
            for last_segment in last_segment_len:
                if segment < total:
                    if last_segment < total:
                        result = randomize_segment_lengths(
                            total, segment, seed=seed)
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


def test_expand_values_by_lengths_into_vector_01():
    """
    Test invalid input:  lists are different lengths
    """
    values = [1, 2, 3]
    lengths = [1]

    with pytest.raises(AssertionError):
        result = expand_values_by_lengths_into_vector(values, lengths)


def test_expand_values_by_lengths_into_vector_02():
    """
    Test empty input
    """
    values = []
    lengths = []
    result = expand_values_by_lengths_into_vector(values, lengths)
    correct_result = []
    assert result == correct_result


def test_expand_values_by_lengths_into_vector_03():
    """
    Test valid input:  one value, one length of zero
    """
    values = [7]
    lengths = [0]
    result = expand_values_by_lengths_into_vector(values, lengths)
    correct_result = []
    assert result == correct_result


def test_expand_values_by_lengths_into_vector_04():
    """
    Test valid input:  one value, one length
    """
    values = [7]
    lengths = [3]
    result = expand_values_by_lengths_into_vector(values, lengths)
    correct_result = [7, 7, 7]
    assert result == correct_result


def test_expand_values_by_lengths_into_vector_05():
    """
    Test valid input:  multiple values and lengths of zero
    """
    values = [1, 2, 3]
    lengths = [0, 0, 0]
    result = expand_values_by_lengths_into_vector(values, lengths)
    correct_result = []
    assert result == correct_result


def test_expand_values_by_lengths_into_vector_06():
    """
    Test valid input:  multiple values and lengths
    """
    values = [1, 2, 3]
    lengths = [1, 1, 1]
    result = expand_values_by_lengths_into_vector(values, lengths)
    correct_result = [1, 2, 3]
    assert result == correct_result


def test_expand_values_by_lengths_into_vector_07():
    """
    Test valid input:  multiple values and lengths
    """
    values = [1, 2, 3]
    lengths = [1, 0, 1]
    result = expand_values_by_lengths_into_vector(values, lengths)
    correct_result = [1, 3]
    assert result == correct_result


def test_expand_values_by_lengths_into_vector_08():
    """
    Test valid input:  multiple values and lengths
    """
    values = [1, 2, 3]
    lengths = [3, 1, 7]
    result = expand_values_by_lengths_into_vector(values, lengths)
    correct_result = [1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3]
    assert result == correct_result


def test_expand_values_by_lengths_into_vector_09():
    """
    Test valid input:  multiple values and lengths
    """
    values = ['a', 'b', 'c']
    lengths = [2, 0, 4]
    result = expand_values_by_lengths_into_vector(values, lengths)
    correct_result = ['a', 'a', 'c', 'c', 'c', 'c']
    assert result == correct_result


def test_expand_values_by_lengths_into_vector_10():
    """
    Test valid input
    """
    values =  np.array([1, 2, 3])
    lengths = np.array([1, 1, 1])
    result = expand_values_by_lengths_into_vector(values, lengths)
    correct_result =  np.array([1, 2, 3])
    np.testing.assert_array_equal(result, correct_result)


def test_expand_values_by_lengths_into_vector_11():
    """
    Test invalid input:  negative length
    """
    values =  np.array([1, 2, 3])
    lengths = np.array([1, -1, 1])

    with pytest.raises(AssertionError):
        result = expand_values_by_lengths_into_vector(values, lengths)


def test_generate_and_combine_trends_01():
    """
    Test valid input
    """

    seed = 827816
    time_n = 10
    trend_n = 1
    last_segment_len = -1
    trend_slope_min = -1 
    trend_slope_max = 1

    result_all = generate_and_combine_trends(
        time_n, trend_n, last_segment_len, trend_slope_min, trend_slope_max, 
        seed)
    result = result_all.time_series

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
    last_segment_len = -1
    trend_slope_min = 3 
    trend_slope_max = 3.03

    result_all = generate_and_combine_trends(
        time_n, trend_n, last_segment_len, trend_slope_min, trend_slope_max, 
        seed)
    result = result_all.time_series

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
    last_segment_len = -1
    trend_slope_min = -8 
    trend_slope_max = -7

    result_all = generate_and_combine_trends(
        time_n, trend_n, last_segment_len, trend_slope_min, trend_slope_max, 
        seed)
    result = result_all.time_series

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


