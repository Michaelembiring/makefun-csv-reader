#!/usr/bin/env python3
"""Fast drop-in CSV reader (csv-reader-speed-01).

read_csv(text, header) -> {"columns", "rows", "errors"}

Semantics (must match csv-reader 1.0.0 exactly):
  - Read left to right; comma = field sep; double quote = quote char.
  - A leading U+FEFF is dropped.
  - A record ends at LF or CRLF. A line left with no characters is no record.
  - Outside quotes, a CR not followed by LF is data.
  - A field opening with a quote runs to the next non-doubled quote; "" -> one
    quote; CRLF / lone CR inside quotes -> LF. Junk after the closing quote up to
    the next comma / line break is appended verbatim (quote_junk). An
    unterminated quote is a literal quote and the rest is re-read unquoted
    (unterminated_quote).
  - A quote inside an unquoted field is a literal character.
  - Fields stay strings; never trimmed or converted.
  - columns = first record when header else None.
  - rows = records after the first (all records when header is false), each
    padded (short_row) or cut (long_row) to the first record's width.
  - errors: {code, line, row} in the order met; row is the record's index in
    rows, None for the header. line is the 1-based line where the record starts,
    counting every LF, CRLF and lone CR inside quotes.

Speed notes:
  Two fast paths keep the common cases in C:
    - Text with no double quote at all is split into records with one compiled
      regex pass (CRLF / CR / LF only) and each record split on commas with
      str.split — no per-field Python loop.
    - A quoted field whose span to the next quote has no CR/LF and no doubled
      quote is a single slice.
  The general per-character scan handles only the genuinely complex fields
  (escaped quotes, embedded newlines, junk after the closing quote).
"""

__all__ = ["read_csv", "main"]

import re as _re

# Matches the next character that ends an unquoted field or a record. Using a
# compiled character class lets CPython scan in one C-level pass, which is much
# faster than walking the string in Python or doing several str.find calls.
# NOTE: a double quote is NOT a separator here — a quote inside an unquoted
# field is a literal character (only a quote that *opens* a field is special).
_SEP = _re.compile("[,\n\r]").search

# Splits text into records on CRLF / CR / LF only (the spec's line boundaries).
# Note: this deliberately does NOT split on other Unicode line boundaries
# (\v, \f, \x85, \u2028, \u2029) — the spec treats those as ordinary data.
_RS = _re.compile("\r\n|\r|\n")


def _read_noquotes(text, header):
    """Fast path for text with no double quote anywhere.

    Splits the text into records on CRLF / CR / LF in one C-level regex pass,
    then splits each record on commas. Line numbers are tracked for errors.
    """
    errors = []
    # Split into records on CRLF / CR / LF (the spec's only line boundaries) in
    # one C-level pass. Each split piece is a record body; an empty piece is a
    # blank line (no record) but still consumed a line break.
    parts = _RS.split(text)
    records = []          # list of (start_line, fields)
    line = 1
    for fields in parts:
        if fields == "":
            # Empty piece: a blank line (no record); it consumed one line break.
            line += 1
            continue
        row = fields.split(",")
        if len(row) == 1 and row[0] == "":
            # Blank line: no record.
            line += 1
            continue
        records.append((line, row))
        line += 1
    return _assemble(records, errors, header)


def _assemble(records, errors, header):
    """Shared tail: drop blank records, split header, normalise row width."""
    if not records:
        return {"columns": None if not header else [], "rows": [], "errors": errors}
    first_fields = records[0][1]
    width = len(first_fields)
    if header:
        columns = first_fields
        data = records[1:]
    else:
        columns = None
        data = records
    rows = []
    for idx, (rline, rfields) in enumerate(data):
        w = len(rfields)
        if w < width:
            errors.append({"code": "short_row", "line": rline, "row": idx,
                           "fields": w})
            rows.append(rfields + [""] * (width - w))
        elif w > width:
            errors.append({"code": "long_row", "line": rline, "row": idx,
                           "extra": rfields[width:]})
            rows.append(rfields[:width])
        else:
            rows.append(rfields)
    return {"columns": columns, "rows": rows, "errors": errors}


def read_csv(text, header):
    """Parse `text` and return {"columns", "rows", "errors"}."""
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    # Strip a single leading BOM (U+FEFF).
    if text[:1] == "\ufeff":
        text = text[1:]

    n = len(text)
    errors = []
    records = []          # list of (start_line, fields)

    # Fast path: records that contain no double quote at all can be split on
    # commas in one C-level call, avoiding the per-field Python loop entirely.
    # Most real CSV is quote-free, so this is the common case.
    if '"' not in text:
        return _read_noquotes(text, header)

    i = 0
    line = 1

    while i < n:
        rec_line = line
        fields = []

        # ---- Parse one record (one or more fields) ----
        while True:
            # ---- Parse one field starting at i ----
            if text[i] == '"':
                # ---- Quoted field ----
                # Fast path: if the span to the next quote has no CR/LF and the
                # quote is not doubled, the field is a single slice. Only fall
                # back to the general char-scan when the field is complex.
                i += 1
                q = text.find('"', i)
                if q == -1:
                    # Unterminated: opening quote literal, rest is unquoted.
                    # The remainder is inside quotes, so CR/LF normalise to LF.
                    rest = text[i:n]
                    if "\r" in rest:
                        rest = rest.replace("\r\n", "\n").replace("\r", "\n")
                        line += rest.count("\n")
                    elif "\n" in rest:
                        line += rest.count("\n")
                    field = '"' + rest
                    errors.append({"code": "unterminated_quote",
                                   "line": rec_line, "row": None})
                    fields.append(field)
                    records.append((rec_line, fields))
                    i = n
                    break
                # Check whether this closing quote is escaped (doubled).
                if q + 1 < n and text[q + 1] == '"':
                    # Doubled quote: fall through to general scan.
                    pass
                else:
                    span = text[i:q]
                    if "\r" not in span and "\n" not in span:
                        # Simple quoted field, no escapes, no newlines.
                        field = span
                        i = q + 1
                        # Trailing junk up to , / LF / CR.
                        junk_start = i
                        m = _SEP(text, i)
                        j = n if m is None else m.start()
                        if j > junk_start:
                            field += text[junk_start:j]
                            errors.append({"code": "quote_junk",
                                           "line": rec_line, "row": None})
                        i = j
                        fields.append(field)
                        if i >= n:
                            records.append((rec_line, fields))
                            break
                        d = text[i]
                        if d == ",":
                            i += 1
                            if i >= n:
                                fields.append("")
                                records.append((rec_line, fields))
                                break
                            continue
                        if d == "\r":
                            if i + 1 < n and text[i + 1] == "\n":
                                i += 2
                            else:
                                i += 1
                            line += 1
                        else:
                            i += 1
                            line += 1
                        records.append((rec_line, fields))
                        break
                # ---- General quoted field (char scan) ----
                # i points just after the opening quote; seg starts here.
                parts = None
                closed = False
                seg = i
                while i < n:
                    d = text[i]
                    if d == '"':
                        i += 1
                        if i < n and text[i] == '"':
                            if parts is None:
                                parts = [text[seg:i - 1] + '"']
                            else:
                                parts.append(text[seg:i - 1] + '"')
                            i += 1
                            seg = i
                            continue
                        closed = True
                        break
                    if d == "\r":
                        if parts is None:
                            parts = [text[seg:i]]
                        else:
                            parts.append(text[seg:i])
                        if i + 1 < n and text[i + 1] == "\n":
                            i += 2
                        else:
                            i += 1
                        line += 1
                        parts.append("\n")
                        seg = i
                        continue
                    if d == "\n":
                        if parts is None:
                            parts = [text[seg:i]]
                        else:
                            parts.append(text[seg:i])
                        i += 1
                        line += 1
                        parts.append("\n")
                        seg = i
                        continue
                    i += 1

                if closed:
                    if parts is None:
                        field = text[seg:i - 1]
                    else:
                        parts.append(text[seg:i - 1])
                        field = "".join(parts)
                    # Trailing junk after the closing quote, up to , / LF / CR.
                    junk_start = i
                    m = _SEP(text, i)
                    j = n if m is None else m.start()
                    if j > junk_start:
                        field += text[junk_start:j]
                        errors.append({"code": "quote_junk", "line": rec_line,
                                       "row": None})
                    i = j
                else:
                    # Unterminated: opening quote is literal; the remainder is
                    # read as unquoted. parts holds the de-escaped content.
                    if parts is None:
                        field = '"' + text[seg:n]
                    else:
                        parts.append(text[seg:n])
                        field = '"' + "".join(parts)
                    errors.append({"code": "unterminated_quote", "line": rec_line,
                                   "row": None})
                    fields.append(field)
                    records.append((rec_line, fields))
                    i = n
                    break               # record ends at EOF

                fields.append(field)
                # After a closed quoted field: comma, line end, or EOF.
                if i >= n:
                    records.append((rec_line, fields))
                    break
                d = text[i]
                if d == ",":
                    i += 1
                    if i >= n:
                        fields.append("")
                        records.append((rec_line, fields))
                        break
                    continue
                # CR / LF -> end of record.
                if d == "\r":
                    if i + 1 < n and text[i + 1] == "\n":
                        i += 2
                    else:
                        i += 1
                    line += 1
                else:
                    i += 1
                    line += 1
                records.append((rec_line, fields))
                break

            # ---- Unquoted field ----
            m = _SEP(text, i)
            j = n if m is None else m.start()
            fields.append(text[i:j])
            i = j
            if i >= n:
                records.append((rec_line, fields))
                break
            d = text[i]
            if d == ",":
                i += 1
                if i >= n:
                    fields.append("")
                    records.append((rec_line, fields))
                    break
                continue
            if d == "\r":
                if i + 1 < n and text[i + 1] == "\n":
                    i += 2
                else:
                    i += 1
                line += 1
            else:
                i += 1
                line += 1
            records.append((rec_line, fields))
            break

    # ---- Drop records that are a single empty field (blank lines) ----
    records = [r for r in records if not (len(r[1]) == 1 and r[1][0] == "")]

    return _assemble(records, errors, header)


def main():
    """CLI: read JSON {text, header} on stdin, write the JSON answer to stdout."""
    import sys
    import json

    data = sys.stdin.buffer.read()
    try:
        obj = json.loads(data.decode("utf-8"))
    except Exception:
        obj = {"text": data.decode("utf-8", "surrogatepass"), "header": False}

    text = obj.get("text", "")
    header = bool(obj.get("header", False))
    result = read_csv(text, header)
    out = json.dumps(result, ensure_ascii=False)
    sys.stdout.buffer.write(out.encode("utf-8", "surrogatepass"))


if __name__ == "__main__":
    main()