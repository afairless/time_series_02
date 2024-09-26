 
from src.tracker_1 import (
    track_global_variables,
    )


def test_track_global_variables_01():
    """
    Test valid input
    """

    code = 'a = 1; a = "meow"; a = -7.4'

    result = track_global_variables(code)
    assert result['a'] == [1, 'meow', -7.4]


def test_track_global_variables_02():
    """
    Test valid input
    """

    code = (
        'a = 1;'
        'a = "meow";'
        'a = -7.4')

    result = track_global_variables(code)
    assert result['a'] == [1, 'meow', -7.4]


def test_track_global_variables_03():
    """
    Test valid input
    """

    code = (
        'a = 1;'
        'b = "meow";'
        'b = 12;'
        'a = "bark";'
        'c = [];'
        'b = -7.4')

    result = track_global_variables(code)
    assert result['a'] == [1, 'bark']
    assert result['b'] == ['meow', 12, -7.4]
    assert result['c'] == [[]]


