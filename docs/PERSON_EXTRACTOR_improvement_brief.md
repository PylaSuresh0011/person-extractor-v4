# PERSON_EXTRACTOR_V4 — Error analysis & improvement brief

**Test document:** `10th eng.pdf` (Telangana SCERT Class X English textbook, 274 pages)
**Reference (gold) list:** 279 distinct person / character names (hand-verified)
**Labelled results:** `comparison.csv` (every extractor entry marked correct / partial / not-a-name, plus all missed gold names)

## 1. Current scores

| Metric | All 425 entries | Accepted only (306) |
|---|---|---|
| Entry precision (entry is a real name) | 59.1% | 70.3% |
| Recall (distinct people found) | 74.9% (209/279) | 64.5% |
| F1 | 63.1% | 65.5% |
| Truncated / broken spans | 52 | 48 |
| Non-names | 122 | 43 |
| Duplicate variants of the same person | 42 | 35 |

The accept/review split only partly works. 14% of accepted entries are not names, and 30% of review entries (36 of 119) are real names.

**Target after the fixes:** precision ≥ 90%, recall ≥ 90%, and fewer than 5 truncated spans.

---

## 2. Failure patterns, ordered by impact

### P1. Page headers, footers and watermarks are read as names (largest single error)
- `English` (329 hits) comes from the footer on every page: *"English Class X  Free distribution by T.S. Government 2019-20"*.
- The vertical watermark letters `SC / ER / T / TE / LA / NG / AN / A` also leak in and glue onto nearby tokens.

**Fix**
- Before NER, find lines that repeat on 20% or more of pages, using a normalised text hash, and delete them.
- Delete lines that contain only 1–2 uppercase letters, using the regex `^\s*[A-Z]{1,2}\s*$`.
- Remove page numbers and running titles.

### P2. Capitalised ordinary words (sentence-initial or instruction words)
- **False positives:** `Read`, `I. Read`, `List`, `Comment`, `Sit`, `Convey`, `I'm`, `Man`, `Day`, `God`, `House`, `Story`, `Reader`, `Verb`, `Humour`, `Surfer`, `Potter`, `Grandpa`, `Amma`, `Lord`, `West`, `North`, `Workbook`, `Phrasal Verbs`, `I. Adverbial Clauses`.
- **Cause:** the model treats any capitalised token as a name candidate.

**Fix**
- Run a **document-level lowercase test**. If a word also appears in lowercase elsewhere in the same document ("read", "list", "man", "story"), reject it as a single-token name unless a title or a strong name context comes before it.
- Add a stoplist for textbook and exercise vocabulary: Read, List, Comment, Discourse, Reader, Workbook, Verb, Clause, Story, Exercise, Glossary, Unit, Answer.
- Reject contractions and pronouns (`I'm`, `You've`), and kinship or role words used alone (`Grandpa`, `Amma`, `Nana`, `Lord`, `Nurse`, `Father`).
- Reject numbered section headers matching `^[IVX]+\.\s`.

### P3. Non-person proper nouns accepted as people
| Type | Examples from this run |
|---|---|
| Places | Mysore, Kashmir, Goa, Calcutta, Madras, Rameswaram, Mahabubnagar, Dirang, Wadgaon, Koko, Bengal, Himalayas, Western Ghats |
| Districts and areas | `Krishna Dist`, `Ranga Reddy` (a district), `Medak Dist`, `Prakasam District`, `Gandhi Nagar`, `Narayanaguda` |
| Buildings and streets | `Philip's Church`, `Buddha Vihar`, `John's Enclave`, `Ravindra Bharathi`, `Mosque Street`, `JILL'S Villa`, `Taj Mahal` |
| Awards and records | Padma Bhushan, Padma Vibhushan, Bharat(h) Ratna, Nobel Prize, Booker, Guinness, Kalaimamani, Golden Jubilee |
| Works (films, books, epics, music) | Maya Bazaar, Mayabazaar, Mahanati, Missamma, Agantuk, Chitrabani, Pater Panchali, Devadasu, Sonam, Varasdar, Ramayana, Mahabharatha, Gita, Missa Solemnis, Dollar Sose |
| Groups, castes, religions, languages | Muslim, Dalit, Mahar, Brahmin, Monpa, Kauravas, Paandavas, Moghuls, Yadava, French |
| Products and brands | Pinto (Ford car), Manohar (publisher), Xavier (college) |

**Fix**
- Classify the entity type from context, not just capitalisation.
- **Place cues:** `in/at/from/to X`, `X Dist`, `X District`, `X Village`, `X Road`, `X Nagar`, `X, India`.
- **Building cues:** the suffix words Church, Street, Vihar, Enclave, Villa, Mahal, College, School, Bharathi (as a venue).
- **Work cues:** the word sits in quotes `'X'`, or comes after `film / novel / book / play / song / movie`, or is followed by a year `(1955)`.
- **Award cues:** `award`, `prize`, `ratna`, `bhushan`, `vibhushan`, `medal`, `record`.
- **Group cues:** plural forms (`-s`, `-as`) and words like caste, community, religion, tribe.
- Drop plural family names (`Slaters`, `Jordans`) or merge them into the person.

### P4. Broken spans: names cut off at line breaks, page breaks or initials (52 cases)
- **Cut by a line or page break:** `Dr. Brahm` (→ Brahm *Prakash*), `Lee De` (→ *Forest*), `Vincent Van` (→ *Gogh*), `Norman Vincent` (→ *Peale*), `Raja Ram Mohan` (→ *Roy*), `Dr. Avul Pakir Jainulabdeen Abdul` (→ *Kalam*), `Abel Merry` (→ *weather*), `Martin Luther` (→ *King Jr.*), `Dr. Vikram` (→ *Sarabhai*).
- **Cut at initials:** `Dr. G`, `Dr. B`, `Dr. K`, `Dr. T`, `Mr. K`, `Mr. T`, `Sri P.`, `Dr. B. Syama` (→ Sundari).
  - **Cause:** the source writes `G.Gopal Reddy` and `K.Venkata`, with no space after the dot, so the tokenizer splits the name in the wrong place.
- **Fragments as standalone names:** `Reddy`, `Rao`, `Sastry`, `Patil`, `Iyer`, `Srinivas`, `Narayana`, `Sudha`, `Murthy`, `Pingali`, `Nandamuri`, `Ludwig`, `Saratchandra`.

**Fix**
- **Text normalisation before NER:**
  - Insert a space after each initial with `re.sub(r'\b([A-Z])\.(?=[A-Z])', r'\1. ', text)`.
  - Join hyphenated line breaks, and join single newlines inside paragraphs.
  - Merge text across page boundaries after the header/footer removal in P1.
- **Span extension:** after NER, extend each span greedily to the right over:
  - capitalised tokens;
  - name particles: `van, von, de, da, di, du, bin, al, la, le`;
  - suffixes: `Jr., Sr., II, III, IV, VI, XI`.
- **Initials regex:** `(?:[A-Z]\.\s?){1,4}\s?[A-Z][a-z]+(?:\s[A-Z][a-z]+)*` should match `Y.G.K. Murthy`, `K.N. Anandan`, `S.V. Ranga Rao`, `M.L. Vasantha Kumari`, `T.T.P. Mohan Babu`.
- **Never emit a span that ends in a bare initial or a title:** reject `Dr. G`, `Mr. K`, `Sri P.`.
- **Surname-only fragments:** emit them only if they resolve to a full name elsewhere in the document (see P6); otherwise send them to review.

### P5. Spans that run too far to the left or right
- **Examples:** `As Bayaji`, `Observing Henry`, `I'm Sarada`, `You've Jill`, `Ray- Ganashatru`, `Dreamer Wake`.

**Fix**
- Strip leading function words, sentence-start adverbs, verbs and conjunctions: As, When, If, Observing, I'm, You've, Dear, Then.
- Split on dashes and hyphens followed by a capital letter.
- Before accepting a span, check that every token in it is name-like.

### P6. No entity resolution, so one person appears several times (42 duplicates)
- **Kalam:** `Kalam`, `Abdul Kalam`, `Dr. Kalam`, `Abul`, `Dr. Avul Pakir Jainulabdeen Abdul`.
- **Amelia Slater:** `Mrs. Slater`, `Amelia`, `Mrs. Amelia Slater`.
- **Abel Merryweather:** `Abel`, `Abel Merryweather` (listed twice), `Mr. Merryweather`, `Abel Merry`.
- **Ambedkar:** `Ambedkar`, `Babasaheb`, `Babasaheb Amebedkar` (a typo in the source).
- **Satyajit Ray:** `Ray`, `Satyajit`, `Satyajit Ray`, `Manikda`.
- **Aunt Jane:** listed twice, once as accepted and once as review.

**Fix**
- Normalise each name: strip honorifics (Mr, Mrs, Ms, Dr, Sri, Smt, Prof, Fr, Sir, Lord, Colonel, Professor), lowercase it, and remove punctuation.
- Cluster names when one name's tokens are a subset of another's (`Kalam` ⊂ `Abdul Kalam`), using the longest form as the canonical name.
- Use a fuzzy match (edit distance ≤ 2) for spelling variants: Bacchan/Bachchan, Shastry/Sastry, Akber/Akbar, Amebedkar/Ambedkar, Hendricks/Hendriks.
- Learn aliases from explicit patterns:
  - `X (Y)`, as in `Nandamuri Taraka Ramarao (NTR)`;
  - `known as X`;
  - `X [as Y was affectionately called]`;
  - `affectionately called X`.
- When a surname maps to two or more people (`Murthy` → Sudha / Narayana / Satyanarayana), do not merge it. Mark it ambiguous.
- Output one row per canonical person, with an `aliases` column and a summed count.

### P7. Missed names (70 people, 25% of the gold list), grouped by pattern
| Missed pattern | Missed examples | Detection rule to add |
|---|---|---|
| Comma-separated lists of names, often with a role in brackets | Relangi, Padmanabham, Madhavapeddi, Pasumarthy, Pitambaram, Chakrapani, Rajababu, Brahmanandam | **List propagation:** if at least 50% of the items in a coordinated list are confirmed names, accept the rest. `Name (music)` and `Name (art)` are strong cues. |
| Acronym names | ANR (NTR was also missed as an acronym) | Accept 2–4 uppercase letters when they sit in a list with people, or after `(…)` that defines an alias. |
| Initials plus surname | K.V. Reddy, S.V. Ranga Rao, L.V. Prasad, Y.G.K. Murthy, K.N. Anandan, T.V. Ramakishan, M.D. Hatknagalekar, P. Satyavathi | The initials regex and dot-spacing fix in P4. |
| Credit pages (title + name + comma + designation) | Navuluri Peraiah, Y.G.K. Murthy, K.N. Anandan, T.V. Ramakishan | `^(Sri|Smt|Dr|Mr|Mrs|Ms|Prof)\.?\s+NAME,` gives a high-confidence person. |
| Generational and regnal suffixes | Henry Ford II, Tom Watson Sr., Pope Paul VI | Allow the suffixes `II|III|Sr\.?|Jr\.?|VI|XI` (see P4). |
| Famous Western names in lists | Stephen Hawking, Walt Disney, Marie Curie, Lee Iacocca, Steve Appel, Xenophon, Aristophanes, Pissarro, Alexander, Birbal | Add a gazetteer of well-known people. A name after `said`, `by`, `wrote` or `was fired by` is a person cue. |
| Groups of people | Wright Brothers | `X Brothers` / `X Sisters` is a person group. Emit it as a person. |
| Kinship and relationship cues | wife **Sumanthamma**, daughters **Rashmitha** and **Ashwitha**, married **Sumathi**, brother-in-law **Jallaluddin**, his wife **Jane Wilde**, Father: **Frank Hawking** | `(wife|husband|son|daughter|brother|sister|cousin|father|mother|married|brother-in-law)\s*:?\s+NAME` gives a person. |
| Dialogue speaker labels | Prudhvi, Sarath, Rishi, Frederick, Albert, Isaac (prompted lines in the play) | `^NAME\s*:` at the start of a line gives a speaker, so a person. |
| Letter sign-offs and salutations | Ramu (`Yours faithfully, Ramu`), Ravi (`Dear Ravi`) | Name after `Yours faithfully/sincerely/lovingly,` or after `Dear`. |
| Exercise sentences with fill-in blanks | Srinu, Sravani, Sharath, Sarath | Names before `____` blanks are missed. Treat runs of underscores as a single placeholder token **before** NER. |
| Mythological and film characters | Subhadra, Parvathi, Apu, Harihar, Mary | A name that is spoken to (`" Subhadra! …"`) or is the possessor of a world (`Apu's world`) is a character. Add a gazetteer of Indian mythology names. |
| Multiple people with the same name | Vithoba of Wadgaon vs Vithoba Ghayakute | Keep them separate when the full forms differ. |

### P8. Accept/review decision is poorly calibrated
- **Accepted, but not names:** Read, Muslim, Mysore, Kashmir, Taj Mahal, Padma Bhushan, Bharata Ratna, Guinness, Jawan, Dalit, Sonam, Pinto.
- **Sent to review, but real names:** Ashiamma, Samsuddin, Socrates, Plato, Mother Teresa, Duryodhana, Nagireddy, Jija Buwa, Parasu Buwa, Kadegaonkar Buwa, Thomas Agonyo, Pothuraju, Ganiraju, Dhanaraju, Sasirekha, Suryakantam, M. Yadaiah, Manikda.

**Fix:** replace the current rule with a scoring model. Positive features:
- a title before the name (Mr, Dr, Sri, Smt);
- a kinship cue;
- a speech verb after it (said, asked, replied);
- a speaker label `Name:`;
- the name appears in a list of names;
- the name has two or more name-like tokens;
- the word never appears in lowercase in the document;
- an Indian or Western first-name gazetteer hit.

Negative features:
- the word also appears in lowercase in the document;
- a place, building or award cue;
- the word is inside quotes as a title;
- the name is part of a header or footer;
- the name is a single token with a high dictionary frequency.

Accept a name at a score of 0.7 or higher, send 0.4–0.7 to review, and drop anything below 0.4.

---

## 3. Suggested pipeline order
1. Extract the PDF text, then **remove headers, footers and watermarks** (P1).
2. **Normalise the text:** add a space after initials, join hyphenated and broken lines, and collapse `____` blanks (P4, P7).
3. **Collect candidates** from NER, the regex patterns (initials, title + name, speaker labels, kinship, sign-offs) and the gazetteers.
4. **Clean the spans:** extend over particles and suffixes, strip leading function words, and reject trailing initials (P4, P5).
5. **Filter by entity type:** place, building, work, award and group cues, plus the lowercase test and the stoplist (P2, P3).
6. **Propagate through lists:** accept the remaining items of mostly-name lists (P7).
7. **Score the names and decide** accept, review or drop (P8).
8. **Resolve entities:** cluster aliases into one canonical name with an `aliases` column (P6).

## 4. Evaluation harness (add this so every change is measured)
- Use `comparison.csv` and the gold list as a fixture: `tests/fixtures/10th_eng_gold.csv`.
- Write `eval.py`. It should normalise each name (strip honorifics, lowercase, remove punctuation) and match it to the gold list by exact match, then alias, then fuzzy match (edit distance ≤ 2).
  - Report precision, recall and F1.
  - Report the number of truncated spans, the number of duplicates, and the top 20 false positives and false negatives.
- Add regression unit tests, one per pattern:
  - `"Sri K.Venkata Rama Rayudu, OiC"` → `K. Venkata Rama Rayudu`.
  - `"Dr. Brahm\n\n254 Free distribution…\nPrakash, father of"` → `Brahm Prakash`.
  - `"English Class X Free distribution by T.S. Government"` → no names.
  - `"Ghantasala (music), M.L.Vasantha Kumari, Leela, Suseela and Madhavapeddi (playback)"` → 5 names.
  - `"His wife Sumanthamma is in a state of shock"` → `Sumanthamma`.
  - `"Lee Iacocca was fired by Henry Ford II"` → `Lee Iacocca` and `Henry Ford II`.
  - `"Read the following passage"` → no names.
  - `"'Maya Bazaar' has been voted…"` → no names.
- Fail the CI run if F1 drops below the previous best.

## 5. Quick wins, ordered by effort vs gain
1. Remove headers, footers and watermarks. This drops `English` (329 hits) and a lot of noise.
2. Add the space after initials and join broken lines. This fixes most of the 52 truncated spans and recovers the initial-style names.
3. Add the lowercase test and the stoplist. This removes most single-word false positives.
4. Add place, building, award and work suffix and context filters.
5. Add alias clustering. This removes the 42 duplicates.
6. Add list propagation and the kinship, speaker and sign-off patterns. This recovers about 30 of the 70 missed names.
