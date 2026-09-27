from __future__ import annotations

import io
import json
import unittest

from PIL import Image, PngImagePlugin

from report_verification import inspect_label_evidence


class LabelEvidenceTest(unittest.TestCase):
    def test_records_metadata_names_without_retaining_values(self) -> None:
        image = Image.new("RGB", (16, 16), "white")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("AIGC", "private-value-must-not-be-retained")
        metadata.add_text("Software", "private-generator-name")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", pnginfo=metadata)

        evidence = inspect_label_evidence(buffer.getvalue())
        serialized = json.dumps(evidence, ensure_ascii=False)

        self.assertEqual(evidence["implicit_label"]["status"], "detected")
        self.assertIn("AIGC", evidence["implicit_label"]["detected_keys"])
        self.assertIn("Software", evidence["generation_metadata_clues"]["detected_keys"])
        self.assertNotIn("private-value-must-not-be-retained", serialized)
        self.assertNotIn("private-generator-name", serialized)


if __name__ == "__main__":
    unittest.main()
