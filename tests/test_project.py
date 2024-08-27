 
# import pytest

from src.project import (
    randomize_segment_lengths,
    flatten_list_of_lists,
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


# def test_dummy_function02_01():
#     """
#     Test input of wrong data type
#     """

#     input = -1

#     with pytest.raises(AssertionError):
#         result = dummy_function02(input)


