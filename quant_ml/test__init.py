from unittest import TestCase
from test import DF_TEST


class TestStrategy(TestCase):

    def test__init(self):
        import quant_ml as qml
        self.assertTrue(hasattr(qml, "indicators"))
        self.assertTrue(hasattr(DF_TEST, "ta_sma"))


