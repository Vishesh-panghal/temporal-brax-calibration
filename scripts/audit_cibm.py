import os
import re

with open('manuscript_cibm/main_cibm.tex', 'r') as f:
    text = f.read()

checks = [
    ('documentclass elsarticle', r'\documentclass[preprint,12pt]{elsarticle}' in text),
    ('linenumbers', r'\linenumbers' in text),
    ('single author Vishesh', 'Vishesh Panghal' in text),
    ('email', 'vishesh@poornima.org' in text),
    ('PIET affiliation', 'Poornima Institute of Engineering and Technology' in text),
    ('highlights environment', r'\begin{highlights}' in text),
    ('graphical abstract environment', r'\begin{graphicalabstract}' in text),
    ('CRediT section', 'credit authorship contribution statement' in text.lower()),
    ('Declaration of competing interest', 'declaration of competing interest' in text.lower()),
    ('Data availability', 'data and code availability' in text.lower()),
    ('Funding declaration', r'\section*{Funding}' in text),
    ('Algorithmic complexity', 'algorithmic and computational complexity' in text.lower()),
    ('Related work', r'\subsection{Related Work}' in text),
    ('Scope and boundary', 'scope and boundary of findings' in text.lower()),
    ('No IEEEPARstart', r'\IEEEPARstart' not in text),
    ('No ieeecolor', 'ieeecolor' not in text),
    ('No cite package', '{cite}' not in text),
    ('No false Zenodo DOI (13824519)', '13824519' not in text),
    ('No private CITI ID (61229341)', '61229341' not in text),
    ('No old 11,946 count', '11,946' not in text and '5,973' not in text),
    ('Tracks all 16,604 evaluations', '16,604' in text),
    ('Tracks 8,302 prospective images', '8,302' in text),
    ('AI declaration includes OpenAI Codex', 'openai codex' in text.lower()),
    ('Supplementary material PDF exists', os.path.exists('manuscript_cibm/supplementary_material.pdf')),
    ('Main PDF exists', os.path.exists('manuscript_cibm/main_cibm.pdf')),
]

print("=== PRE-SUBMISSION AUTOMATED CHECKLIST ===")
all_passed = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"[{status}] {name}")

# Abstract word count check
abs_start = text.find(r'\begin{abstract}')
abs_end = text.find(r'\end{abstract}')
abs_text = text[abs_start + len(r'\begin{abstract}'):abs_end].strip()
# Remove TeX macros for word counting
clean_abs = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^\}]*\})*', ' ', abs_text)
clean_abs = re.sub(r'[\$\{\}\\]', ' ', clean_abs)
abs_words = clean_abs.split()
abs_count = len(abs_words)
abs_ok = abs_count <= 250
print(f"\n=== ABSTRACT WORD COUNT AUDIT (Limit: 250 words) ===")
print(f"[{'PASS' if abs_ok else 'FAIL'}] Word count: {abs_count} words (limit: 250)")

hl_start = text.find(r'\begin{highlights}')
hl_end = text.find(r'\end{highlights}')
hl_block = text[hl_start:hl_end]
items = re.findall(r'\\item\s+([^\n]+)', hl_block)

print("\n=== HIGHLIGHTS LENGTH AUDIT (Limit: 85 characters) ===")
hl_ok = True
for i, item in enumerate(items, 1):
    cleaned = item.strip()
    l = len(cleaned)
    status = "OK" if l <= 85 else "EXCEEDS"
    if l > 85:
        hl_ok = False
    print(f"Bullet {i} ({l} chars) [{status}]: {cleaned}")

overclaims = ['safe automation', 'safe rule-out', 'definitively disprove', 'completely remediates', 'genuine temporal discrimination degradation', 'genuine data drift']
print("\n=== CLINICAL AND TEMPORAL OVERCLAIMS PURGE AUDIT ===")
oc_ok = True
for oc in overclaims:
    cnt = len(re.findall(re.escape(oc), text, re.IGNORECASE))
    status = "PASS (0 occurrences)" if cnt == 0 else f"FAIL ({cnt} found)"
    if cnt > 0:
        oc_ok = False
    print(f"[{status}] Term: '{oc}'")

# BibTeX check
with open('manuscript_cibm/references.bib', 'r') as f:
    bib = f.read()
bib_entries = re.findall(r'@\w+\{([^,]+),', bib)
print(f"\nTotal BibTeX Entries in references.bib: {len(bib_entries)}")

# Specific BibTeX audit
brax_ok = 'reis2022brax' in bib and '487' in bib and '10.1038/s41597-022-01608-8' in bib
finlayson_ok = 'finlayson2021clinician' in bib and '10.1056/NEJMc2104626' in bib
mimic_ok = 'johnson2019mimic' in bib and 'johnson2019mimicjpg' in bib
print(f"[PASS if brax_ok else FAIL] BRAX citation verified (487, Sci Data 2022): {brax_ok}")
print(f"[PASS if finlayson_ok else FAIL] Finlayson citation verified (NEJM 2021): {finlayson_ok}")
print(f"[PASS if mimic_ok else FAIL] MIMIC-CXR and MIMIC-CXR-JPG dual citations verified: {mimic_ok}")

bib_ok = brax_ok and finlayson_ok and mimic_ok

print("\n=== OVERALL AUDIT SUMMARY ===")
if all_passed and hl_ok and oc_ok and abs_ok and bib_ok:
    print("ALL AUDIT CHECKS PASSED: 100% Ready for CIBM Submission.")
else:
    print("SOME CHECKS FAILED - REVIEW REQUIRED.")
