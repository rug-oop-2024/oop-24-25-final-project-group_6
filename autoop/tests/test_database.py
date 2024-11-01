import unittest

import random
import tempfile
import os
import sys


current_dir = os.path.dirname(os.path.realpath(__file__))
functional_dir = os.path.dirname(current_dir)
auto_oop_dir = os.path.dirname(functional_dir)
sys.path.insert(0, auto_oop_dir)


from autoop.core.database import Database
from autoop.core.storage import LocalStorage


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.storage = LocalStorage(tempfile.mkdtemp())
        self.db = Database(self.storage)

    def test_init(self):
        self.assertIsInstance(self.db, Database)

    def test_set(self):
        id = str(random.randint(0, 100))
        entry = {"key": random.randint(0, 100)}
        self.db.set("collection", id, entry)
        self.assertEqual(self.db.get("collection", id)["key"], entry["key"])

    def test_delete(self):
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        self.db.delete("collection", id)
        self.assertIsNone(self.db.get("collection", id))
        self.db.refresh()
        self.assertIsNone(self.db.get("collection", id))

    def test_persistance(self):
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        other_db = Database(self.storage)
        self.assertEqual(other_db.get("collection", id)["key"], value["key"])

    def test_refresh(self):
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        other_db = Database(self.storage)
        self.db.set("collection", key, value)
        other_db.refresh()
        self.assertEqual(other_db.get("collection", key)["key"], value["key"])

    def test_list(self):
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", key, value)
        # collection should now contain the key
        self.assertIn((key, value), self.db.list("collection"))

if __name__ == "__main__":
    unittest.main()
