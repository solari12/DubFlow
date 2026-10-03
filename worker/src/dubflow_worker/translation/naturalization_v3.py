from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NaturalizationResult:
    text: str
    rules_applied: tuple[str, ...]
    review_reason: str | None


class ContextAwareVietnameseNaturalizer:
    """Conservative spoken-Vietnamese surface cleanup for the v6 validation path."""

    name = "context-aware-vi-rules-v3"
    _rule_specs = (
        ("duration_expression", re.compile(r"\btôi đã học (.+?) trong (một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|\d+) năm\b", re.I), r"Tôi đã học \1 được \2 năm"),
        ("spoken_first_step", re.compile(r"\bđiều đầu tiên (?:bạn )?muốn làm là\b", re.I), "Trước tiên, bạn nên"),
        ("writing_system_punctuation", re.compile(r"\bHiragana\s+Katakana\b", re.I), "Hiragana, Katakana"),
        ("writing_system_term", re.compile(r"\bba hệ thống viết chính\b", re.I), "ba hệ chữ viết chính"),
        ("spoken_how_to", re.compile(r"\bhãy để tôi dạy các bạn làm thế nào để làm nó chính xác\b", re.I), "để tôi chỉ các bạn cách làm chính xác"),
        ("download_loanword", re.compile(r"\bdownload\b", re.I), "tải về"),
        ("first_app_action", re.compile(r"\bđiều đầu tiên tôi làm là ứng dụng\b", re.I), "việc đầu tiên tôi làm là dùng ứng dụng"),
        ("remove_literal_thinker", re.compile(r"\btôi nghĩ rằng\b", re.I), "tôi nghĩ"),
        ("natural_sense_check", re.compile(r"\bnếu điều đó có ý nghĩa\b", re.I), "nếu bạn hiểu ý tôi"),
        ("self_study_expression", re.compile(r"\bhọc cách học\b", re.I), "học cách tự học"),
        ("app_learning_expression", re.compile(r"\b(?:nên )?hãy học rằng khi bạn sử dụng (?:một )?ứng dụng\b", re.I), "hãy học dần qua ứng dụng"),
        ("learning_transition", re.compile(r"\b(?:và )?sau đó khi bạn tiếp tục học\b", re.I), "khi tiếp tục học"),
        ("example_sentence_expression", re.compile(r"\b(?:hãy )?tạo ra các câu ví dụ\b", re.I), "hãy đặt câu ví dụ"),
        ("apply_current_learning", re.compile(r",?\s*như sử dụng những gì bạn đang học\b", re.I), " và áp dụng những gì bạn đang học"),
        ("grammar_confusion", re.compile(r"\bngữ pháp,?\s+(?:là )?(?:một chút )?nhầm lẫn\b", re.I), "ngữ pháp hơi khó hiểu"),
        ("grammar_confusion", re.compile(r"\bngữ pháp,?\s+(?:để )?(?:là )?(?:một chút )?nhầm lẫn\b", re.I), "ngữ pháp hơi khó hiểu"),
        ("grammar_confusion", re.compile(r"\bđể là một chút nhầm lẫn\b", re.I), "hơi khó hiểu"),
        ("resource_recommendation", re.compile(r"\bđó là nơi tôi khuyên bạn nên\b", re.I), "đó là lúc tôi khuyên bạn nên"),
        ("guide_recommendation", re.compile(r"\btriển khai (?=[\w'’]+(?:['’]s)?\s+Guide\b)", re.I), "tham khảo "),
        ("guide_recommendation", re.compile(r"\bthực hiện (?=[\w'’]+(?:['’]s)?\s+Guide\b)", re.I), "tham khảo "),
        ("free_resource", re.compile(r"\bđó là nguồn(?: lực)? miễn phí(?: này)?\b", re.I), "đây là tài liệu miễn phí"),
        ("use_resource", re.compile(r"\bvà sử dụng (?:nó|đó),? học\b", re.I), "Hãy dùng tài liệu đó để học"),
        ("spoken_recommendation", re.compile(r"\bvì vậy chọn\b", re.I), "Vậy, hãy chọn"),
        ("spoken_early_transition", re.compile(r"\b(?:và )?sớm(?: thôi)? bạn sẽ thấy\b", re.I), "Rồi bạn sẽ thấy"),
    )
    _ambiguous_patterns = (
        (re.compile(r"\bblog\s+và\s+lỗi\s+tốt\b", re.I), "Ambiguous source phrase was left literal for review."),
    )
    _source_conditions = {
        "duration_expression": re.compile(r"\b(?:study|studying|learn|learning)\b.*\bfor\s+(?:\w+|\d+)\s+years?\b", re.I),
        "spoken_first_step": re.compile(r"\bfirst thing\b.*\bwant to do\b", re.I),
        "writing_system_punctuation": re.compile(r"\bHiragana\b.*\bKatakana\b", re.I),
        "writing_system_term": re.compile(r"\bthree main writing systems\b", re.I),
        "spoken_how_to": re.compile(r"\blet me teach\b.*\bhow to do\b", re.I),
        "download_loanword": re.compile(r"\bApp Store\b|\bdownload\b|ダロノード|ダウンロード", re.I),
        "first_app_action": re.compile(r"\bfirst thing\b.*\bapp\b|最初.{0,15}アプリ", re.I),
        "remove_literal_thinker": re.compile(r"\bI think\b|思う|思って", re.I),
        "natural_sense_check": re.compile(r"\bif that makes sense\b", re.I),
        "self_study_expression": re.compile(r"\blearn how to learn\b|\bself.study\b", re.I),
        "app_learning_expression": re.compile(r"\buse an app\b", re.I),
        "learning_transition": re.compile(r"\bkeep learning\b", re.I),
        "example_sentence_expression": re.compile(r"\bcreate example sentences\b", re.I),
        "apply_current_learning": re.compile(r"\buse what you(?:'|’)re learning\b", re.I),
        "grammar_confusion": re.compile(r"\bconfus(?:e|ing|ion)\b", re.I),
        "resource_recommendation": re.compile(r"\bThat's where I recommend\b", re.I),
        "guide_recommendation": re.compile(r"\b(?:guide|resource)\b", re.I),
        "free_resource": re.compile(r"\bfree resource\b", re.I),
        "use_resource": re.compile(r"\busing that\b", re.I),
        "spoken_recommendation": re.compile(r"\bpick one\b", re.I),
        "spoken_early_transition": re.compile(r"\bsoon enough you'll find\b", re.I),
    }

    def naturalize_with_report(self, text: str, *, language: str, source_context: str) -> NaturalizationResult:
        if language.split("-", 1)[0].lower() != "vi":
            return NaturalizationResult(text, (), "This naturalizer only supports Vietnamese output.")
        result = text
        applied: list[str] = []
        for rule_name, pattern, replacement in self._rule_specs:
            context_pattern = self._source_conditions.get(rule_name)
            if context_pattern is not None and not context_pattern.search(source_context):
                continue
            result, count = pattern.subn(replacement, result)
            if count:
                applied.extend([rule_name] * count)

        # Repair duplicated conjunctions and punctuation spacing without changing clauses.
        result, count = re.subn(r"\b(?:và\s+){2,}", "và ", result, flags=re.I)
        if count:
            applied.append("duplicate_conjunction")
        result, count = re.subn(r"\s+([,.;!?])", r"\1", result)
        if count:
            applied.append("punctuation_spacing")
        result, count = re.subn(r"(?<=[^\W\d_])\s+(?=(?:Đó là|Đây là|Hãy dùng)\b)", ". ", result, flags=re.I)
        if count:
            applied.append("sentence_boundary")
        result, count = re.subn(r"([.!?])\s+([a-zà-ỹ])", lambda match: match.group(1) + " " + match.group(2).upper(), result)
        if count:
            applied.append("sentence_capitalization")
        result, count = re.subn(r"([,;:])(?=\S)", r"\1 ", result)
        if count:
            applied.append("punctuation_spacing")
        result, count = re.subn(r"[ \t]{2,}", " ", result)
        if count:
            applied.append("whitespace")
        result = result.strip()
        if result:
            result = result[0].upper() + result[1:]

        review_reason = None
        for pattern, reason in self._ambiguous_patterns:
            if pattern.search(result):
                review_reason = reason
                break
        if source_context and not applied:
            review_reason = review_reason or "No safe spoken-Vietnamese rewrite matched; review the literal output."
        return NaturalizationResult(result, tuple(applied), review_reason)

    def naturalize(self, text: str, *, language: str, source_context: str) -> str:
        return self.naturalize_with_report(text, language=language, source_context=source_context).text
