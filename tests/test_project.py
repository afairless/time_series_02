 
# import pytest

from src.project import (
    flatten_list_of_lists,
    )


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


