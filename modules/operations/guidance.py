"""
modules/operations/guidance.py

CAPTURE GUIDANCE SPRINT — TICKET-CG-01/02/03: Turkish rewrite & Coaching integration.

Verilen video kaydından ve rekonstrüksiyon sonuçlarından elde edilen teknik metrikleri,
operatör için anlamlı ve eyleme dökülebilir Türkçe talimatlara dönüştürür.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from modules.shared_contracts.models import (
    CaptureGuidance, GuidanceSeverity, AssetStatus
)

# ─────────────────────────────────────────────────────────────────────────────
# Severity helpers
# ─────────────────────────────────────────────────────────────────────────────

_INFO = GuidanceSeverity.INFO
_WARN = GuidanceSeverity.WARNING
_CRIT = GuidanceSeverity.CRITICAL

def _msg(code: str, message: str, severity: GuidanceSeverity) -> Dict[str, Any]:
    return {"code": code, "message": message, "severity": severity}

# ─────────────────────────────────────────────────────────────────────────────
# Message Registry - Turkish Localized
# ─────────────────────────────────────────────────────────────────────────────

_COACHING_MESSAGES = {
    # System / Backend
    "SYSTEM_FAILURE_CONFIG": "Sistem yapılandırma hatası. Lütfen teknik ekiple iletişime geçin.",
    "SYSTEM_FAILURE_PIPELINE": "Beklenmedik bir işlem hatası oluştu. Lütfen tekrar deneyin veya destek isteyin.",
    
    # Generic Recapture
    "RECAPTURE_NEEDED": "Bu çekim kalite standartlarını karşılamıyor. Lütfen aşağıdaki talimatları takip ederek yeniden çekim yapın.",
    
    # Coverage / Geometric (CG-01)
    "ORBIT_GAP_LEFT": "Objenin sol tarafı yeterince kapsanmamış; nesnenin etrafında daha tam bir tur çek.",
    "ORBIT_GAP_RIGHT": "Objenin sağ arka tarafı yeterince görünmüyor; nesnenin etrafında daha tam bir tur çek.",
    "LOW_HORIZONTAL_COVERAGE": "Yatay kapsama yetersiz. Objenin etrafında tam bir daire çizdiğinizden emin olun.",
    "WEAK_ORBIT_CONTINUITY": "Açı geçişleri çok kopuk; daha yumuşak ve sürekli bir yörüngeyle yeniden çek.",
    "INSUFFICIENT_VIEWPOINT_SPREAD": "Bakış açısı çeşitliliği çok dar. Objenin etrafında daha geniş bir yay çizerek hareket edin.",
    "RECAPTURE_LOW_DIVERSITY": "Yeterli farklı bakış açısı yakalanamadı. Objenin etrafında daha yavaş ve farklı açılardan dönün.",
    "RECAPTURE_NARROW_MOTION": "Kamera hareketi çok dar. Objenin etrafında daha belirgin ve geniş bir yay çizerek çekim yapın.",
    "RECAPTURE_SCALE_FLAT": "Çekimler hep aynı yükseklikten yapılmış. Farklı yüksekliklerden (yukarıdan 30-60 derece) de görüntü alın.",
    "RECAPTURE_TOO_FEW_FRAMES": "Kullanılabilir kare sayısı çok az. Kamerayı daha yavaş hareket ettirin ve netliğe odaklanın.",
    "RECAPTURE_LOW_CONFIDENCE": "Nesne net olarak seçilemiyor. Arka planın sade ve ışığın homojen olduğundan emin olun.",
    "MISSING_TOP_VIEWS": "Üst açılar eksik. Objenin üst kısmını görmek için kamerayı yukarıdan aşağıya (45 derece) eğerek bir tur daha atın.",
    
    # Reconstruction Informed (CG-03)
    "LOW_RECONSTRUCTABLE_OVERLAP": "Görüntüler arası örtüşme çok az; kareler birbirine bağlanamıyor. Daha yavaş ve kesintisiz hareket edin.",
    "INSUFFICIENT_POINTS": "Yetersiz görsel detay. Objenin üzerinde odaklanın ve kamera sarsıntısını azaltın.",
    "SUBJECT_TOO_SMALL": "Nesne bazı bölümlerde çok küçük kalmış; kamerayı nesneye biraz daha yaklaştırın (kenarları kesmeden).",

    # Processing status
    "AWAITING_UPLOAD": "Video yüklenmesi bekleniyor. Yükleme tamamlandığında işlem otomatik başlayacaktır.",
    "PROCESSING_RECONSTRUCTION": "Kareler ayrıştırıldı. 3D model oluşturma işlemi devam ediyor.",
    "PROCESSING_GENERIC": "İşlem devam ediyor. Sistem otomatik olarak bir sonraki aşamaya geçecektir.",
    "READY_FOR_REVIEW": "Model oluşturuldu. Lütfen dashboard üzerinden görsel kalite kontrolü yapın.",
    "READY_FOR_PUBLISH": "Model kalite kontrolünden geçti ve yayınlanmaya hazır.",
}

# Mapping patterns to internal codes
_FAILURE_PATTERNS = [
    ("not configured",      "SYSTEM_FAILURE_CONFIG",  _CRIT),
    ("CUDA",                "SYSTEM_FAILURE_CONFIG",  _CRIT),
    ("binary not found",    "SYSTEM_FAILURE_CONFIG",  _CRIT),
    
    # Diagnostic codes from Coverage/Geometric
    ("gap_left",            "ORBIT_GAP_LEFT",         _CRIT),
    ("gap_right",           "ORBIT_GAP_RIGHT",        _CRIT),
    ("ORBIT_GAP_LEFT",      "ORBIT_GAP_LEFT",         _CRIT),
    ("ORBIT_GAP_RIGHT",     "ORBIT_GAP_RIGHT",        _CRIT),
    ("LOW_HORIZONTAL",      "LOW_HORIZONTAL_COVERAGE",_CRIT),
    
    # Coaching variants (soft)
    ("COACHING",            "ORBIT_GAP_LEFT",         _WARN),
    
    ("CONTINUITY",          "WEAK_ORBIT_CONTINUITY",  _WARN),
    ("SPREAD",              "INSUFFICIENT_VIEWPOINT_SPREAD", _WARN),
    
    ("viewpoint diversity", "RECAPTURE_LOW_DIVERSITY",_CRIT),
    ("object motion",       "RECAPTURE_NARROW_MOTION",_WARN),
    ("scale/shape",         "RECAPTURE_SCALE_FLAT",   _WARN),
    ("readable",            "RECAPTURE_TOO_FEW_FRAMES",_WARN),
    
    # Reconstruction signals
    ("insufficient size",   "LOW_RECONSTRUCTABLE_OVERLAP", _CRIT),
    ("model too small",     "LOW_RECONSTRUCTABLE_OVERLAP", _CRIT),
    ("registered_images",   "LOW_RECONSTRUCTABLE_OVERLAP", _CRIT),
]

def _match_failure_reason(reason: str) -> Optional[Dict[str, Any]]:
    if not reason:
        return None
    reason_lower = reason.lower()
    for pattern, code, severity in _FAILURE_PATTERNS:
        if pattern.lower() in reason_lower or code.lower() in reason_lower:
            return _msg(code, _COACHING_MESSAGES.get(code, code), severity)
    return None

class GuidanceAggregator:
    """
    Operator kılavuzunu (Guidance) Türkçe olarak oluşturur.
    Hem çekim aşaması metriklerini hem de rekonstrüksiyon sonuçlarını harmanlar.
    """

    def generate_guidance(
        self,
        session_id: str,
        status: AssetStatus,
        coverage_report: Optional[Dict[str, Any]] = None,
        validation_report: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None,
        reconstruction_stats: Optional[Dict[str, Any]] = None,
    ) -> CaptureGuidance:

        messages: List[Dict[str, Any]] = []
        should_recapture = False
        is_ready_for_review = False
        next_action = "İşlem yapılıyor — Lütfen bekleyin."

        # 1. Status Based Base Message
        if status == AssetStatus.CREATED:
            next_action = "İşlemin başlaması için ürün videosunu yükleyin."
            messages.append(_msg("AWAITING_UPLOAD", _COACHING_MESSAGES["AWAITING_UPLOAD"], _INFO))

        elif status == AssetStatus.CAPTURED:
            next_action = "Model oluşturma devam ediyor. Bekleyin."
            messages.append(_msg("PROCESSING_RECONSTRUCTION", _COACHING_MESSAGES["PROCESSING_RECONSTRUCTION"], _INFO))

        elif status == AssetStatus.RECAPTURE_REQUIRED:
            next_action = "Yeniden çekim gerekiyor. Lütfen aşağıdaki talimatları inceleyin."
            should_recapture = True
            messages.append(_msg("RECAPTURE_NEEDED", _COACHING_MESSAGES["RECAPTURE_NEEDED"], _CRIT))
            
            detail = _match_failure_reason(failure_reason or "")
            if detail:
                messages.append(detail)

        elif status == AssetStatus.VALIDATED:
            next_action = "Model hazır. Lütfen onaylayın."
            is_ready_for_review = True
            messages.append(_msg("READY_FOR_REVIEW", _COACHING_MESSAGES["READY_FOR_REVIEW"], _INFO))

        # 2. Coverage / Geometric Enrichment
        if coverage_report:
            enriched = self._enrich_from_coverage(coverage_report)
            messages.extend(enriched)
            if coverage_report.get("overall_status") != "sufficient":
                should_recapture = True

        # 3. Reconstruction Informed Feedback
        if reconstruction_stats:
            reg = reconstruction_stats.get("registered_images", 0)
            total = len(reconstruction_stats.get("input_frames", [])) or 1
            ratio = reg / total
            
            if status == AssetStatus.RECAPTURE_REQUIRED or ratio < 0.6:
                if ratio < 0.5 and reg > 0:
                    messages.append(_msg(
                        "LOW_RECONSTRUCTABLE_OVERLAP", 
                        _COACHING_MESSAGES["LOW_RECONSTRUCTABLE_OVERLAP"], 
                        _CRIT
                    ))
                    should_recapture = True

        # 4. Deduplication
        seen = set()
        deduped = []
        for m in messages:
            if m["code"] not in seen:
                seen.add(m["code"])
                deduped.append(m)
        messages = deduped

        # 5. Localized next_action mapping
        if should_recapture:
            next_action = "Yeniden çekim gerekli: Talimatları izleyerek yeni bir video gönderin."
        elif is_ready_for_review:
            next_action = "Dashboard'u açarak 3D modeli inceleyin ve onaylayın."

        return CaptureGuidance(
            session_id=session_id,
            status=status,
            next_action=next_action,
            should_recapture=should_recapture,
            is_ready_for_review=is_ready_for_review,
            messages=messages,
            coverage_summary=coverage_report,
            validation_summary=validation_report,
        )

    def _enrich_from_coverage(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = []
        reasons = report.get("reasons", [])
        
        for r in reasons:
            detail = _match_failure_reason(r)
            if detail:
                messages.append(detail)
                
        # Top down view
        if not report.get("top_down_captured", True):
            messages.append(_msg("MISSING_TOP_VIEWS", _COACHING_MESSAGES["MISSING_TOP_VIEWS"], _WARN))
            
        return messages

    def to_markdown(self, guidance: CaptureGuidance) -> str:
        lines = [
            f"# Çekim Kılavuzu - {guidance.session_id}",
            f"**Durum:** `{guidance.status.value.upper()}`",
            f"**Sonraki Adım:** {guidance.next_action}",
            "",
        ]

        fatals = [m for m in guidance.messages if m["severity"] == GuidanceSeverity.CRITICAL]
        coaching = [m for m in guidance.messages if m["severity"] in {GuidanceSeverity.WARNING, GuidanceSeverity.INFO}]

        if fatals:
            lines.append("## 🛑 KRİTİK ENGELLER (Yeniden Çekim Gerekli)")
            for msg in fatals:
                lines.append(f"- **{msg['message']}**")
            lines.append("")

        if coaching:
            lines.append("## 💡 GELİŞTİRME ÖNERİLERİ (Coaching)")
            for msg in coaching:
                label = "Öneri" if msg["severity"] == GuidanceSeverity.INFO else "Uyarı"
                lines.append(f"- *{label}:* {msg['message']}")
            lines.append("")

        if not fatals and not coaching:
            lines.append("## ℹ️ Durum")
            lines.append("- İşlem normal seyrinde devam ediyor. Özel bir talimat bulunmamaktadır.")

        lines.append("")
        return "\n".join(lines)
