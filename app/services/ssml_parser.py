"""SSML parser for multi-speaker dialogue and prosody control.

Supports a subset of SSML tags:
  <speak> — root element
  <prosody rate="slow" pitch="+2st" volume="loud"> — post-processing params
  <break time="500ms"/> — insert silence
  <emphasis level="strong"> — mapped to instruct text
  <voice name="Vivian"> — speaker switch for multi-speaker dialogue
"""
import re
import xml.etree.ElementTree as ET


def _parse_time_ms(time_str):
    """Parse a time string like '500ms' or '1s' into milliseconds."""
    time_str = time_str.strip().lower()
    if time_str.endswith('ms'):
        return int(time_str[:-2])
    elif time_str.endswith('s'):
        return int(float(time_str[:-1]) * 1000)
    try:
        return int(time_str)
    except ValueError:
        return 500  # default


def _parse_pitch(pitch_str):
    """Parse pitch string like '+2st', '-3st', 'high', 'low'."""
    pitch_str = pitch_str.strip().lower()
    pitch_map = {
        'x-low': -6, 'low': -3, 'medium': 0,
        'high': 3, 'x-high': 6, 'default': 0,
    }
    if pitch_str in pitch_map:
        return pitch_map[pitch_str]
    # Parse semitone notation
    match = re.match(r'([+-]?\d+(?:\.\d+)?)\s*st', pitch_str)
    if match:
        return float(match.group(1))
    return 0


def _parse_rate(rate_str):
    """Parse rate string like 'slow', 'fast', '1.5', '80%'."""
    rate_str = rate_str.strip().lower()
    rate_map = {
        'x-slow': 0.5, 'slow': 0.75, 'medium': 1.0,
        'fast': 1.25, 'x-fast': 1.5, 'default': 1.0,
    }
    if rate_str in rate_map:
        return rate_map[rate_str]
    if rate_str.endswith('%'):
        return float(rate_str[:-1]) / 100.0
    try:
        return float(rate_str)
    except ValueError:
        return 1.0


def _parse_volume(vol_str):
    """Parse volume string like 'loud', 'soft', '+6dB'."""
    vol_str = vol_str.strip().lower()
    vol_map = {
        'silent': -40, 'x-soft': -24, 'soft': -20,
        'medium': -16, 'loud': -12, 'x-loud': -8, 'default': -16,
    }
    if vol_str in vol_map:
        return vol_map[vol_str]
    return -16


def _emphasis_to_instruct(level):
    """Map emphasis level to an instruct text hint."""
    level = level.strip().lower()
    mapping = {
        'strong': 'speak with strong emphasis and conviction',
        'moderate': 'speak with moderate emphasis',
        'reduced': 'speak softly and understated',
        'none': '',
    }
    return mapping.get(level, 'speak with emphasis')


def _process_element(element, context):
    """Recursively process an SSML element and extract segments."""
    segments = []
    tag = element.tag.lower() if isinstance(element.tag, str) else ''

    if tag == 'break':
        time_str = element.get('time', '500ms')
        ms = _parse_time_ms(time_str)
        segments.append({
            'type': 'break',
            'duration_ms': ms,
        })
        return segments

    if tag == 'voice':
        context = {**context, 'speaker': element.get('name', context.get('speaker'))}

    if tag == 'prosody':
        prosody = {**context.get('prosody', {})}
        if element.get('rate'):
            prosody['speed'] = _parse_rate(element.get('rate'))
        if element.get('pitch'):
            prosody['pitch_shift'] = _parse_pitch(element.get('pitch'))
        if element.get('volume'):
            prosody['volume_normalize'] = _parse_volume(element.get('volume'))
        context = {**context, 'prosody': prosody}

    if tag == 'emphasis':
        level = element.get('level', 'moderate')
        instruct = _emphasis_to_instruct(level)
        if instruct:
            context = {**context, 'instruct': instruct}

    # Process text content
    if element.text and element.text.strip():
        segments.append({
            'type': 'speech',
            'text': element.text.strip(),
            'speaker': context.get('speaker'),
            'language': context.get('language'),
            'instruct': context.get('instruct'),
            'prosody': context.get('prosody', {}),
        })

    # Process children
    for child in element:
        segments.extend(_process_element(child, context))
        # Text after child element (tail text)
        if child.tail and child.tail.strip():
            segments.append({
                'type': 'speech',
                'text': child.tail.strip(),
                'speaker': context.get('speaker'),
                'language': context.get('language'),
                'instruct': context.get('instruct'),
                'prosody': context.get('prosody', {}),
            })

    return segments


def parse_ssml(ssml_text):
    """Parse SSML text into a list of segments.

    Returns list of dicts:
      - {'type': 'speech', 'text': ..., 'speaker': ..., 'instruct': ..., 'prosody': {...}}
      - {'type': 'break', 'duration_ms': ...}
    """
    # Wrap in <speak> if not already
    ssml_text = ssml_text.strip()
    if not ssml_text.lower().startswith('<speak'):
        ssml_text = f'<speak>{ssml_text}</speak>'

    try:
        root = ET.fromstring(ssml_text)
    except ET.ParseError as e:
        raise ValueError(f"Invalid SSML: {e}") from e

    context = {
        'speaker': None,
        'language': None,
        'instruct': None,
        'prosody': {},
    }

    return _process_element(root, context)
