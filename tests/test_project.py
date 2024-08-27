 
import pytest

from src.project import (
    dummy_function01,
    dummy_function02,
    )


def test_dummy_function01_01():
    """
    Test valid input
    """

    result = dummy_function01()
    correct_result = 1
    assert result == correct_result


def test_dummy_function02_01():
    """
    Test input of wrong data type
    """

    input = -1

    with pytest.raises(AssertionError):
        result = dummy_function02(input)


def test_dummy_function02_02():
    """
    Test valid input
    """

    input = 1

    result = dummy_function02(input)
    correct_result = 2
    assert result == correct_result
