from __future__ import annotations

import json

from stt.schemas import TranscriptionResult, TranscriptionSegment


def test_transcription_result_is_json_serializable():
    result = TranscriptionResult(
        recognition_language="th-TH",
        status="completed",
        segments=[
            TranscriptionSegment(
                segment_id=1,
                speaker_id="Guest-1",
                text="ข้อความสังเคราะห์",
                recognition_status="recognized",
                offset_seconds=0.0,
                duration_seconds=1.0,
            )
        ],
    )
    payload = result.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)
    assert '"speaker_id": "Guest-1"' in encoded
    assert payload["segments"][0]["segment_id"] == 1


def test_speaker_label_is_neutral_data_only():
    segment = TranscriptionSegment(
        segment_id=1,
        speaker_id="Guest-2",
        text="synthetic",
        recognition_status="recognized",
    )
    assert set(segment.to_dict()) == {
        "segment_id",
        "speaker_id",
        "text",
        "recognition_status",
        "offset_seconds",
        "duration_seconds",
    }
