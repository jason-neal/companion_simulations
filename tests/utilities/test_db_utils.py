from mingle.utilities.db_utils import DBExtractor
import pytest

class FakeTable():
    pass

@pytest.fixture()
def test_table():
    table = FakeTable()
    table.c = None
    return table

class Test_DBExtractor():

    @pytest.mark.parametrize("limit", [5, 50, 10000])
    def test_BDExtractor_initalized(self, test_table, limit):
        x = DBExtractor(table=test_table, limit=limit)
        assert x.limit == limit
        assert isinstance(x, DBExtractor)

