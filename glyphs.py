
CHARACTERS = [
    # 0: virama, alone. A tiny mark upscaled to 32x32, so its images look like blobs
    ("\u0D4D", "virama"),
    # 1-10 vowel signs
    ("\u0D3E", "aa"), ("\u0D3F", "i"), ("\u0D40", "ii"), ("\u0D41", "u"), ("\u0D42", "uu"),
    ("\u0D44", "vocalic r"), ("\u0D46", "e"), ("\u0D47", "ee"), ("\u0D4C", "au"),
    ("\u0D02", "anusvara"),
    # 11-18 independent vowels
    ("\u0D05", "a"), ("\u0D06", "aa"), ("\u0D07", "i"), ("\u0D09", "u"), ("\u0D0B", "vocalic r"),
    ("\u0D0E", "e"), ("\u0D0F", "ee"), ("\u0D12", "o"),
    # 19-54 consonants
    ("\u0D15", "ka"), ("\u0D16", "kha"), ("\u0D17", "ga"), ("\u0D18", "gha"), ("\u0D19", "nga"),
    ("\u0D1A", "cha"), ("\u0D1B", "chha"), ("\u0D1C", "ja"), ("\u0D1D", "jha"), ("\u0D1E", "nya"),
    ("\u0D1F", "tta"), ("\u0D20", "ttha"), ("\u0D21", "dda"), ("\u0D22", "ddha"), ("\u0D23", "nna"),
    ("\u0D24", "ta"), ("\u0D25", "tha"), ("\u0D26", "da"), ("\u0D27", "dha"), ("\u0D28", "na"),
    ("\u0D2A", "pa"), ("\u0D2B", "pha"), ("\u0D2C", "ba"), ("\u0D2D", "bha"), ("\u0D2E", "ma"),
    ("\u0D2F", "ya"), ("\u0D30", "ra"), ("\u0D31", "rra"), ("\u0D32", "la"), ("\u0D33", "lla"),
    ("\u0D34", "zha"), ("\u0D35", "va"), ("\u0D36", "sha"), ("\u0D37", "ssa"), ("\u0D38", "sa"),
    ("\u0D39", "ha"),
    # 55-59 chillus. 55 is the one the anchors leave over
    ("\u0D7B", "chillu n"), ("\u0D7A", "chillu nn"), ("\u0D7C", "chillu r"),
    ("\u0D7D", "chillu l"), ("\u0D7E", "chillu ll"),
    # 60-81 conjuncts, in the paper's Table 1 order
    ("\u0D15\u0D4D\u0D15", "kka"), ("\u0D15\u0D4D\u0D37", "ksha"), ("\u0D19\u0D4D\u0D15", "nka"),
    ("\u0D19\u0D4D\u0D19", "nnga"), ("\u0D1A\u0D4D\u0D1A", "chcha"), ("\u0D1E\u0D4D\u0D1A", "ncha"),
    ("\u0D1E\u0D4D\u0D1E", "nnya"), ("\u0D1F\u0D4D\u0D1F", "tta"), ("\u0D23\u0D4D\u0D1F", "nta"),
    ("\u0D23\u0D4D\u0D23", "nna"), ("\u0D24\u0D4D\u0D24", "ttha"), ("\u0D26\u0D4D\u0D27", "ddha"),
    ("\u0D28\u0D4D\u0D24", "ntha"), ("\u0D28\u0D4D\u0D26", "nda"), ("\u0D28\u0D4D\u0D28", "nna"),
    ("\u0D2A\u0D4D\u0D2A", "ppa"), ("\u0D2C\u0D4D\u0D2C", "bba"), ("\u0D2E\u0D4D\u0D2E", "mma"),
    ("\u0D2F\u0D4D\u0D2F", "yya"), ("\u0D32\u0D4D\u0D32", "lla"), ("\u0D33\u0D4D\u0D33", "llla"),
    ("\u0D35\u0D4D\u0D35", "vva"),
    # 82-84 consonant signs
    ("\u0D4D\u0D2F", "ya sign"), ("\u0D4D\u0D30", "ra sign"), ("\u0D4D\u0D35", "va sign"),
]
assert len(CHARACTERS) == 85, len(CHARACTERS)
