import unittest
from unittest.mock import patch

from app import create_app


class HarvestPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.client = cls.app.test_client()
        cls.options = cls.client.get("/api/options").get_json()

    def find_valid_record(self):
        for crop in self.options["crops"]:
            regions = self.client.get(
                "/api/options", query_string={"crop": crop}
            ).get_json()["regions"]
            for region in regions:
                # The test below uses a known in-season date only after fetching the record
                # indirectly through a calendar prediction candidate.
                for month in range(1, 13):
                    for day in (5, 15, 25):
                        response = self.client.post("/api/predict", json={
                            "crop": crop,
                            "region": region,
                            "planting_date": f"2026-{month:02d}-{day:02d}",
                        })
                        if response.status_code == 200:
                            return crop, region, f"2026-{month:02d}-{day:02d}", response
        self.fail("No valid in-season crop/region/date combination found")

    def test_options_are_filtered_to_south_africa(self):
        self.assertEqual(self.options["crops"], sorted(self.options["crops"]))
        self.assertGreater(len(self.options["crops"]), 0)
        regional = self.client.get(
            "/api/options", query_string={"crop": self.options["crops"][0]}
        ).get_json()
        self.assertGreater(len(regional["regions"]), 0)

    def test_prediction_includes_calendar_and_model_metadata(self):
        _, _, _, response = self.find_valid_record()
        result = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(result["country"], "South Africa")
        self.assertRegex(result["calendar_code"], r"^\d{1,2}_\d{1,2}_\d{1,2}_\d{1,2}$")
        self.assertIn("start", result["harvest_window"])
        self.assertIn("model_available", result)

    def test_out_of_season_planting_returns_calendar_window(self):
        crop, region, _, _ = self.find_valid_record()
        responses = [
            self.client.post("/api/predict", json={
                "crop": crop,
                "region": region,
                "planting_date": f"2026-{month:02d}-{day:02d}",
            })
            for month in range(1, 13)
            for day in (5, 15, 25)
        ]
        out_of_season = next((response for response in responses if response.status_code == 422), None)
        self.assertIsNotNone(out_of_season, "Expected at least one date outside the planting season")
        self.assertIn("planting_window", out_of_season.get_json())

    def test_invalid_requests_are_rejected(self):
        self.assertEqual(self.client.post("/api/predict", json={}).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json={
            "crop": "not-a-crop",
            "region": "not-a-region",
            "planting_date": "2026-01-01",
        }).status_code, 404)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    @patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"})
    @patch("google.genai.Client")
    def test_analysis_keeps_gemini_client_open_until_request_finishes(self, client_factory):
        gemini_client = client_factory.return_value.__enter__.return_value
        gemini_client.models.generate_content.return_value.text = "Test analysis"
        response = self.client.post("/api/analysis", json={
            "crop": "Maize",
            "region": "Gauteng",
            "planting_date_label": "12 October 2026",
            "calendar_code": "28_33_10_14",
            "harvest_window": {
                "start_label": "01 January 2027",
                "end_label": "11 February 2027",
            },
        })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"analysis": "Test analysis"})
        client_factory.assert_called_once_with(api_key="test-key")
        gemini_client.models.generate_content.assert_called_once()
        client_factory.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
