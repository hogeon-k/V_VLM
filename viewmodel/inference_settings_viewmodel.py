from __future__ import annotations

from config.detector_settings import (
    DetectorSettings,
    backend_detail_text,
    backend_display_text,
    load_detector_settings,
    save_detector_settings,
    validate_tensorrt_settings,
)


class InferenceSettingsViewModel:
    def get_detector_settings(self) -> DetectorSettings:
        return load_detector_settings()

    def save_detector_settings(self, detector_settings: DetectorSettings) -> None:
        if detector_settings.detector_backend == "tensorrt":
            validate_tensorrt_settings(detector_settings)
        save_detector_settings(detector_settings)

    def validate_detector_settings(self, detector_settings: DetectorSettings) -> None:
        if detector_settings.detector_backend == "tensorrt":
            validate_tensorrt_settings(detector_settings)

    def backend_display_text(self) -> str:
        return backend_display_text(self.get_detector_settings())

    def backend_detail_text(self) -> str:
        return backend_detail_text(self.get_detector_settings())
