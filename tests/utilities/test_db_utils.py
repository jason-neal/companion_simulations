import pytest

from mingle.utilities.db_utils import DBExtractor


class metadata():
    pass

class FakeTable():
    def __init__(self):
        self.metadata = metadata()
        self.metadata.bind = 5

@pytest.fixture()
def test_table():
    table = FakeTable()
    table.c = {"label1": 0, "label2": 5}
    table.metadata.bind = 5
    return table


class Test_DBExtractor():

    def test_BDExtractor_initalized(self, test_table):
        x = DBExtractor(table=test_table)
        assert isinstance(x, DBExtractor)
        assert x.table == test_table
        assert x.cols == test_table.c

