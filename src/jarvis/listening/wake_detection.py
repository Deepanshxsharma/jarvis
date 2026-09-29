"""Wake word and stop command detection logic."""

from typing import List, Optional
import difflib
import re

from ..debug import debug_log


def is_wake_word_detected(text_lower: str, wake_word: str, aliases: List[str], fuzzy_ratio: float = 0.78) -> bool:
    """
    Check if text contains wake word using exact and fuzzy matching.
    
    Args:
        text_lower: Lowercase text to check
        wake_word: Primary wake word
        aliases: List of wake word aliases
        fuzzy_ratio: Threshold for fuzzy matching (0.0-1.0)
    
    Returns:
        True if wake word detected
    """
    if not text_lower or not text_lower.strip():
        return False
    
    # Combine wake word and aliases
    all_aliases = set(aliases) | {wake_word}
    
    # Check exact match first
    if wake_word in text_lower:
        return True
    
    # Check aliases exact match
    for alias in aliases:
        if alias in text_lower:
            return True
    
    # Fuzzy matching for close variations
    try:
        heard_tokens = [t.strip(".,!?;:()[]{}\"'`).-_/") for t in text_lower.split() if t.strip()]
        for token in heard_tokens:
            for alias in all_aliases:
                ratio = difflib.SequenceMatcher(a=alias, b=token).ratio()
                if ratio >= fuzzy_ratio:
                    debug_log(f"wake word fuzzy match: '{alias}' ~ '{token}' (ratio: {ratio:.3f})", "wake")
                    return True
    except Exception:
        pass
    
    return False


def extract_query_after_wake(text_lower: str, wake_word: str, aliases: List[str]) -> str:
    """
    Extract the query portion after removing wake word.
    
    Args:
        text_lower: Lowercase text containing wake word
        wake_word: Primary wake word
        aliases: List of wake word aliases
    
    Returns:
        Query text with wake word removed
    """
    if not text_lower:
        return ""
    
    all_aliases = set(aliases) | {wake_word}
    fragment = text_lower
    
    # Remove all aliases from the text
    for alias in all_aliases:
        fragment = fragment.replace(alias, " ")
    
    # Clean up punctuation that might be left after wake word removal
    fragment = fragment.strip().lstrip(",.!?;:")
    fragment = fragment.strip()
    
    return fragment if fragment else ""


def _contains_phrase(text_lower: str, phrase: str) -> bool:
    """True when *phrase* appears in *text_lower* as whole words, ignoring punctuation."""
    words = re.sub(r"[^\w\s']", " ", text_lower).split()
    phrase_words = re.sub(r"[^\w\s']", " ", phrase.lower()).split()
    return bool(phrase_words) and f" {' '.join(phrase_words)} " in f" {' '.join(words)} "


def is_interrupt_command(text_lower: str, interrupt_commands: List[str]) -> bool:
    """Check if text contains an interrupt phrase as whole words.

    Whole-word matching keeps "actually" from firing inside "factually" and
    "wait" inside "waiter".
    """
    if not text_lower or not text_lower.strip():
        return False
    for phrase in interrupt_commands:
        if _contains_phrase(text_lower, phrase):
            debug_log(f"interrupt command detected: {phrase} in '{text_lower}'", "voice")
            return True
    return False


def is_stop_command(text_lower: str, stop_commands: List[str], fuzzy_ratio: float = 0.8) -> bool:
    """
    Check if text contains a stop command as whole words.
    
    Args:
        text_lower: Lowercase text to check
        stop_commands: List of stop command phrases
        fuzzy_ratio: Threshold for fuzzy matching short inputs
    
    Returns:
        True if stop command detected
    """
    if not text_lower or not text_lower.strip():
        return False
    
    # Whole-word matches, so "stop" does not fire inside "stopwatch" or a
    # short phrase like "bas" inside "based" when Jarvis's own voice leaks in.
    detected_commands = [cmd for cmd in stop_commands if _contains_phrase(text_lower, cmd)]
    
    # Check fuzzy matches for short inputs (2 words or less)
    if len(text_lower.split()) <= 2:
        try:
            for word in text_lower.split():
                for cmd in stop_commands:
                    ratio = difflib.SequenceMatcher(a=cmd, b=word).ratio()
                    if ratio >= fuzzy_ratio:
                        detected_commands.append(f"{cmd}~{word}")
        except Exception:
            pass
    
    if detected_commands:
        debug_log(f"stop command detected: {detected_commands[0]} in '{text_lower}'", "voice")
        return True
    
    return False
