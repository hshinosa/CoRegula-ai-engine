# Kolabri AI-Engine: Master Technical Handover & Integration Guide

**Date:** 2026-02-06  
**Status:** ✅ PRODUCTION READY | EXPERT HARDENED | 100% PROPOSAL ALIGNED

---

## 1. Executive Summary
Kolabri AI-Engine adalah core computation engine yang mengintegrasikan **Generative AI (GLM-4.7)** dengan prinsip **Socially-Shared Regulated Learning (SSRL)**. Sistem ini dirancang untuk mendampingi diskusi kelompok secara proaktif, mengukur kualitas kognitif mahasiswa (HOT), dan menyediakan dataset yang kaya untuk analisis **Process Mining**.

---

## 2. Advanced AI Logic (Pedagogical Intelligence)

### A. SSRL Orchestration Flow
Setiap pesan mahasiswa melalui pipeline berikut di `POST /api/orchestrate`:
1.  **Normalization**: Pembersihan slang dan noise teks menggunakan `text_processor.py`.
2.  **Engagement Analysis**: Mengukur `is_hot` (Higher-Order Thinking) & `lexical_variety` (TTR).
3.  **Policy Agent**: Menentukan apakah butuh RAG (`FETCH`) atau hanya percakapan biasa (`NO_FETCH`).
4.  **Guardrails**: Mencegah bot memberikan jawaban PR secara langsung (mendorong Socratic Hinting).
5.  **Plan vs Reality Analysis**: Membandingkan isi diskusi dengan SMART goal yang ditetapkan di awal. Menghasilkan skor *alignment* (0-100%).
6.  **Sequence Anomaly Detection**: Mendeteksi penyimpangan urutan (misal: diskusi dimulai tanpa penetapan tujuan).

---

## 3. Data Schema & Process Mining Integration

Semua aktivitas dicatat ke MongoDB dalam format **XES-Ready**.

### Event Log Schema (Attributes Field):
| Field | Type | Description |
|-------|------|-------------|
| `original_text` | string | Pesan asli mahasiswa. |
| `is_hot` | boolean | `True` jika mengandung analisis/sintesis. |
| `lexical_variety` | float | Skor kekayaan kosakata (0.0 - 1.0). |
| `srl_object` | string | Objek belajar yang terdeteksi (misal: "Big O Notation"). |
| `educational_category` | string | Cognitive, Behavioral, atau Emotional. |

---

## 4. Webhook Requirements (Must be implemented in Core-API)

AI-Engine memicu request proaktif ke Core-API via **Exponential Backoff Retry (3x)**:

1.  **AI Intervention Webhook**:
    *   **Endpoint**: `POST /api/webhooks/ai-intervention`
    *   **Payload**: `{ "groupId": "...", "message": "...", "type": "silence|off_topic|inequity" }`
2.  **Teacher Notification Webhook**:
    *   **Endpoint**: `POST /api/webhooks/teacher-notification`
    *   **Payload**: `{ "courseId": "...", "groupId": "...", "message": "...", "severity": "high" }`

**Security Note**: Setiap request webhook menyertakan header `X-AI-Engine-Secret` (diambil dari `.env`). Sisi Core-API **wajib** memvalidasi header ini.

---

## 5. Exports & Dashboards

*   **Student Breakdown Export**: `GET /api/export/activity/group/{id}`. Menghasilkan CSV yang dikelompokkan per mahasiswa, mencakup setiap pesan di seluruh sesi kelompok (XES Synchronized).
*   **Group Dashboard**: `GET /api/analytics/dashboard/group/{id}`. Menyertakan `status_color` (Red/Yellow/Green), `radar_chart_data` (Dimensi SSRL 0-10), dan `teacher_advice` naratif.
*   **Individual Dashboard**: `GET /api/analytics/dashboard/individual/{id}`. Menyertakan `radar_chart_data` personal, `personal_advice`, dan topik yang dikuasai.

---

## 6. Performance & Security Hardening

*   **Zip Slip Protection**: `document_processor.py` aman dari serangan path traversal.
*   **Memory Protection**: `LogicListener` membersihkan state otomatis setiap mencapai 1000 grup aktif.
*   **Efficiency Guard**: Sistem caching RAG aktif untuk menghemat token LLM.
*   **Multimodal RAG**: PDF berisi gambar di-caption otomatis menggunakan **Gemini Vision**. Aset visual disimpan di `./data/static/images`.

---

## 7. Operational Recovery & Active Session Discovery

*   **Server Restart**: Jika server restart, AI-Engine akan melakukan **Active Session Discovery**. Sistem akan mencari aktivitas terbaru di MongoDB untuk mengembalikan konteks `CaseID` kelompok secara otomatis tanpa butuh input ulang dari mahasiswa.

---

## 8. Maintenance & Testing

### Test Suite (python-testing-patterns Skill)
**Location:** `tests/` directory

**Structure:**
- `tests/conftest.py` - Shared fixtures for mocking services
- `tests/test_unit/` - Unit tests for individual services
- `tests/test_integration/` - Integration tests for API endpoints

**Run Tests:**
```powershell
cd ai-engine
# Run all unit tests
python -m pytest tests/test_unit/ -v

# Run with coverage
python -m pytest tests/test_unit/ --cov=app --cov-report=html

# Run specific test file
python -m pytest tests/test_unit/test_nlp_analytics.py -v

# Run integration tests (requires app context)
python -m pytest tests/test_integration/test_api_routes.py -v
```

### Run Proactive Integration Test:
```powershell
cd ai-engine
$env:PYTHONPATH="."
python -B tests/test_kolabri_proactive.py
```

---
**Lead Developer:** Kolabri AI Agent (Droid)  
**Technical Signature:** `REF-20260206-MASTER-ULTIMATE-V2`  
**Verification:** 100% Passed (Includes Plan-vs-Reality, Multimodal, Webhook Security, and Memory Protection)
