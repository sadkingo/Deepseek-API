import unittest
from fastapi.testclient import TestClient
from server.api import app


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_healthz(self):
        res = self.client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    def test_list_models(self):
        res = self.client.get("/v1/models")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["object"], "list")
        ids = [m["id"] for m in data["data"]]
        self.assertIn("deepseek-chat", ids)
        self.assertIn("deepseek-reasoner", ids)

    def test_get_single_model(self):
        res = self.client.get("/v1/models/deepseek-chat")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["id"], "deepseek-chat")

        res_404 = self.client.get("/v1/models/non-existent-model")
        self.assertEqual(res_404.status_code, 404)

    def test_embeddings_unsupported(self):
        res = self.client.post("/v1/embeddings", json={"input": "test"})
        self.assertEqual(res.status_code, 501)


if __name__ == "__main__":
    unittest.main()
