# PERSON_EXTRACTOR: error analysis, round 2 (Social Studies book)

**Test document:** `10th social em.pdf` (Telangana SCERT Class X Social Studies, 346 pages, two-column layout, lots of geography and history)
**Extractor run:** full mode, LLM reviewer on (llama3.1), 487 entries (280 accepted, 207 review), 2m 46s
**Reference (gold) list:** 234 distinct people (`10th_social_gold.csv`)
**Labelled results:** `comparison_social.csv` (every extractor row marked as correct, partial or not-a-name, plus the 27 missed people)

## 1. Scores, compared with round 1

| Metric | Round 1, English book | Round 2, Social book |
|---|---|---|
| Recall, all rows (distinct people found) | 74.9% | **82.9%** ✅ |
| Recall counting partial names | 78.9% | 88.5% |
| Precision, all rows | 59.1% | **49.3%** ❌ |
| Precision, accepted rows only | 70.3% | 73.2% |
| Recall, accepted rows only | 64.5% | 72.6% |
| F1, accepted rows only | 65.5% | **71.0%** |
| Duplicate rows for the same person | 42 | 46 ❌ |
| Partial or truncated rows | 52 | 36 |
| Non-name rows | 122 | 211 ❌ |
| Real names in "review" | 36 of 119 | 35 of 207 (17%) |

**What improved.** The footer and watermark noise is gone. Credit pages now work: 53 of 57 credit names were found. Titled names (Sri, Smt, Dr, Prof) are captured well, and recall is up overall.

**What got worse.** This book is full of rivers, places, months, parties and movements. Many of them look like Indian first names, so non-names more than doubled. The review bucket has turned into a dumping ground: 172 of its 207 rows are junk. Alias groups are computed but the rows are never merged.

---

## 2. New failure patterns in this book

### N1. Rivers, mountains and places that look like person names (high impact, most are ACCEPTED)
- **Accepted as people:** `Ravi` (the river, 9 mentions), `Krishna` (the river and district, 8), `Godavari`, `Mahanadi`, `Cauveri`, `Tapati`, `Teesta`, `Lohit`, `Alakananda`, `Aroma Konda` (a peak), `Vindhya Himachala`, `Nirmal` (a district), `Sujatha Nagar` (a village), `Hussain` (Hussain Sagar lake), `Somnath`, `Malaya`, `Aila` (a cyclone).
- **Cause:** the first-name list ("name list" in `entity_reason`) fires on Ravi, Krishna and Godavari, and nothing in the context checks it.

**Fix (context beats the gazetteer):**
- Use these as negative cues within ±3 tokens: `the X`, `X river`, `river(s) X`, `tributaries`, `flows`, `basin`, `delta`, `hills`, `range`, `peak`, `valley`, `plateau`, `lake`, `sagar`, `cyclone`, `district`, `Dt`, `Dist`, `Nagar`, `mandal`, `village`.
- **List context:** if a name sits in a coordinated list where the other items are known rivers or places ("Jhelum, Chenab, Ravi, Beas and Sutlej"), it is a place.
- **Doc-level majority vote:** if most mentions of X are in geographic context, the whole entity is not a person. One "Ravi" in an exercise must not outweigh nine river mentions.

### N2. Calendar words, months and seasons
- **Examples:** `Karthika`, `Sharad Aswayuja`, `Varsha Sraavana`, `Vasantha Chaitra`, `Hemanta Margashira`, `Shishira Magha`, `Gregorian`, `Julian`.

**Fix:**
- Add a stoplist of the Indian lunar months (Chaitra, Vaishakha, Jyeshtha, Ashadha, Shravana, Bhadrapada, Ashwayuja, Kartika, Margashira, Pushya, Magha, Phalguna).
- Add a stoplist of the six seasons (Vasantha, Grishma, Varsha, Sharad, Hemanta, Shishira).
- Treat any word that comes before "calendar" as not a person.

### N3. Organisations, parties, movements, programmes and companies
- **Examples:** `Meira Paibi` (a women's movement, accepted at 0.85), `Jan Sangh`, `Shiv Sena`, `Sena`, `Janata Dal`, `Bharatiya Lok Dal`, `Mandal` (the Mandal Commission), `Operation Barga`, `Narmada Bachao Andolan`, `Sakala Janula Samme`, `Sagara Haram`, `Telangana Aikya Vedika`, `Tata` (Tata Motors), `Lorenz` (a factory), `Sundaram Fasteners`, `Azim Premji` (a university), `Harvard Crimson`.

**Fix:**
- Treat a word as an organisation when the next word is one of: Movement, Andolan, Sena, Dal, Sangh, Party, Samiti, Sabha, Mahasabha, Commission, Committee, Motors, Foods, Factory, University, College, Airport, Accords, Yatra, Samme.
- If any mention of X is followed by an org head-noun ("Meira Paibi Movement", "Mandal Commission", "Tata Motors", "Azim Premji University"), lower X's person score across the whole document.
- A person name used as an institution name (`Rajiv Gandhi International Airport`, `Kempegouda International Airport`, `Chanda Kantaiah Memorial College`) is not a mention of that person.

### N4. Titles or roles alone, accepted as people
- **Examples:** `Mr. President`, `Maharaja`, `Sri`, `Maulana`, `Acharya`, `Archduke`, `Fuhrer I`, `Tailor` (an occupation), `Retd`.

**Fix:** a span made only of title or role words is never a person. Keep a list of them: President, Maharaja, Raja, Rani, Sri, Smt, Maulana, Acharya, Archduke, Fuhrer, Viceroy, Chancellor, Governor, Pastor, General, Tsar.

### N5. Names written with initials but no dots are missed
- **Missed:** `V P Singh` / `VP Singh`, `IK Gujral`, `AB Vajpayee` / `A.B. Vajpayee`, `KJ Joy`, `M S Khan`, `JM Keynes`.
- **Found only partly:** `L K Advani` → `Advani`, `M G Ramachandran` → `Ramachandran`, `N T Rama Rao` → `Rama Rao`.
- **Leftover fragments:** `Singh`, `Khan` and `Rao` end up as standalone rows.

**Fix:** add a pattern for 1–3 uppercase initials (with or without dots or spaces) followed by a capitalised name:
```
\b(?:[A-Z]\.?\s?){1,3}(?<![A-Z]{4})\s?[A-Z][a-z]+(?:\s[A-Z][a-z]+)?
```
- Accept the 2–3 letter uppercase form (`VP`, `IK`, `AB`, `JM`, `KJ`) only when it comes **directly** before a capitalised name.
- Exclude known acronyms such as `US`, `UN`, `SC`, `PG`, `RSS`, `TDP`, `PGT`.

### N6. A gazetteer conflict deletes a real person
- `K.V. Ranga Reddy` (a Telangana leader) was **missed**, while `Ranga Reddy` / `Rangareddy` (the district) appear as review rows.

**Fix:**
- Initials before a name (`K.V. Ranga Reddy`) are a strong person signal. They must override a place match on the remaining tokens.
- Decide per mention, not per string.

### N7. Asian names, hyphenated names and names in a "Name of Country" pattern
- **Missed:** `Ho Chi Minh`, `Sun Yat-sen` (written `Sun-Yat-Sen`), `Ngo Dinh Diem`, `Nhat Linh`, `Sukarno` ("Sukarno of Indonesia").
- **Fragments output instead:** `Bien`, `Dai`, `Dang`, `Nguyen`, `Pan`, `Chiang Kaishek Environmental`.

**Fix:**
- Allow sequences of 2–4 short capitalised tokens (2–5 letters each) as one name. Vietnamese and Chinese names have this shape.
- Treat hyphens inside a capitalised sequence (`Sun-Yat-Sen`, `Chiang-Kai-Shek`) as part of the name.
- Pattern: `NAME of <Country>`, as in "Nasser of Egypt, Sukarno of Indonesia, Tito of Yugoslavia". When the right-hand side is a country, this gives a person, usually a leader.
- Pattern: `<role> X`, where the role is leader, President, dictator, Emperor, novelist, writer, poet, historian, economist or scientist.

### N8. Two-column layout and figure captions break names
- **Example:** `"General Yahya Lal Bahadur Shastri, in a rally"`. The left column (General Yahya Khan) is interleaved with the right column (Lal Bahadur Shastri).
- **Example:** `"by using the emperor, Bao Fig 13.5 : Ho Chi Minh Dai"`. A figure caption is inserted in the middle of "Bao Dai".
- **Example:** `"Dr. Martin Lutherkinggiving"`. Words from two layout blocks are glued together.

**Fix:**
- Extract text **column by column**, using pdfplumber or PyMuPDF words with x-coordinate clustering, instead of reading whole page lines.
- Find caption lines (`^Fig\.? ?\d+(\.\d+)?\s*:`) and take them out of the running text before NER. Then look for names inside the captions separately: captions are rich in names ("Fig 19.5: Yuri Gagarin").
- Add a sanity check: a name containing a known full name of another person (`Yahya Lal Bahadur Shastri`) gets split.

### N9. Designations glued onto names (over-extension on credit pages)
- **Examples:** `Dr. I. Thirumali Senior Fellow`, `Dr. Ramani Atkuri Medical Practitioner`, `Rao Yerdoor Bangalore`, `Ken Saro Wiwa Peasant`.
- **Cause:** there is no comma between the name and the designation or place.

**Fix:** stop the span at these words:
- designations: Senior, Fellow, Medical, Practitioner, Lecturer, Professor, Director, Consultant, Editor, Teacher, Peasant, Farmer;
- place names, checked with the place filter from N1.

### N10. Alias groups are computed but rows are not merged (46 duplicates)
- `Nehru` and `Jawaharlal Nehru` are two separate rows, and each lists the other as an alias.
- The same happens with Gandhi / Mahatma Gandhi, Indira / Indira Gandhi, Bose / Subhash / Subhash Bose / Subhash Chandra Bose, and Meira / Meira Paibi.
- Some groups are wrong: `West` is an alias of `West Asia`, and `King` is grouped with Dr. Martin Luther King, which is fine, but `King Mahendra` must stay a separate person.

**Fix:**
- After clustering, **emit exactly one row per cluster**: the longest full name, with summed counts and all variants in `aliases`.
- Apply the same accept/review/drop decision to the whole cluster, never to each variant.
- Never cluster a title-led form (`King Mahendra`) with a surname-only form (`King`) when they refer to different people.

### N11. Accept/review decisions are inconsistent and miscalibrated
- **The same score gives different decisions.** `Ravi` (a river) is accepted at 0.65, but `Hitler` (20 mentions) is put in review at 0.65.
  - `Mao Zedong`, `Churchill`, `Mussolini`, `Gorbachev`, `Mountbatten`, `Jinnah`, `Bismarck`, `Woodrow Wilson`, `U Thant` and `Tsar Nicholas II` all land in **review**.
  - `Godavari`, `Mr. President`, `Karthika` and `Tata` are all **accepted**.
- **Review is 83% junk** (172 of 207). Rows with a score of 0.15–0.30 (`West` at 0.15, `Coastal Andhra`, `Food Securty`, `Dual Polity`, `Preamble`, `Sex Ratio`) should be **dropped**, not reviewed.
- **Likely cause:** the LLM reviewer overrides the score without a rule, or the threshold is applied before the reviewer adjusts the score.

**Fix:**
- Make the decision a pure function of the final score: accept at 0.7 or above, review at 0.45–0.7, drop below 0.45. Log any LLM override with its reason.
- Give the LLM reviewer the **sentence context** of 2–3 mentions, not just the string, and ask it to classify the entity as PERSON / PLACE / ORG / WORK / EVENT / TIME / OTHER. Use few-shot examples taken from this book's errors (Ravi the river, Meira Paibi the movement, Karthika the month, Mr. President the title).
- Add a famous-person signal. An entity with many mentions plus a person-context verb (said, led, became, was killed, ruled) should be accepted even if it has only one token (`Hitler`, `Stalin`, `Bismarck`).

### N12. Other missed people
| Missed | Why | Rule |
|---|---|---|
| `Pastor Niemoeller` | "Pastor" is not in the title list | Add religious titles: Pastor, Father, Rev., Bishop, Swami, Guru. |
| `Robert Ley` (spelled "Robert Lay") | Common words ("Lay") are filtered | Allow a lowercase-ambiguous surname after a known first name. |
| `Malcolm X` (spelled "Malcom X") | A one-letter surname is rejected | Allow the `Name X` form. |
| `George Orwell`, `JM Keynes` | Context: "wrote", "economists like" | Treat `wrote` / `argued` / `like NAME` as person cues. |
| `Jesus Christ` | Religious figure | Add a gazetteer of religious figures. |
| `Kumbai`, `Raja Panto` | Deity names in a list | List propagation: the rest of a list of deities is accepted. |
| `Mahalakshmi` (spelled "Mahalakshimi") | Kinship cue: "His parents were Mahalakshimi and Lakshmikanth Rao" | Treat `parents were X and Y` as two persons. |
| `Swathi`, `Ali (an NRI)` | Names in a table cell | Find names in table cells (column header "Name"). |
| `Bhindranwale`, `Sukarno` | Single-token names | Use a person-context cue ("the leader of", "X of Indonesia"). |
| `Pydimarri Venkata Subba Rao`, `Homen Thangjam`, `KJ Joy`, `K. Bhagya Lakshmi`, `M.S. Khan` | Pledge author line and acknowledgement paragraph; `Dr.Homen` has no space | Apply the dot-space fix to titles (`Dr.Homen` → `Dr. Homen`), and look for names in acknowledgement paragraphs and in author lines that start with a dash. |

---

## 3. Rules from round 1 that still hold
- Remove headers, footers and watermarks, and add a space after initials. Both are working now; keep them.
- Use the lowercase test and the stoplist. Extend the stoplist with N2 and N4.
- Separate person vs place vs organisation vs work by context. Extend it with N1 and N3.
- Merge aliases. This is still broken: see N10.

## 4. Testing
- Use **both** gold sets: `10th_eng_gold.csv` (279 names) and `10th_social_gold.csv` (234 names). Each has the columns `canonical_name, aliases, category`.
- `eval.py` should report precision, recall and F1 **per book and macro-averaged**, overall and for accepted rows only. It should also report the duplicate count, the truncated count and the review-bucket junk rate.
- A change is accepted only if it improves the macro-averaged F1 **and** does not lower either book's F1 by more than 1 point. This stops the model being tuned to one book.
- **New unit tests:**
  - `"the Jhelum, the Chenab, the Ravi, the Beas and the Sutlej"` → no persons.
  - `"Vasantha Chaitra, Varsha Sraavana, Sharad Aswayuja"` → no persons.
  - `"Meira Paibi Movement"` → no persons.
  - `"National Front Government with Deve Gowda and IK Gujral as PMs"` → `Deve Gowda`, `I.K. Gujral`.
  - `"NDA government led by AB Vajpayee"` → `A.B. Vajpayee`.
  - `"Nasser of Egypt, Sukarno of Indonesia, Tito of Yugoslavia"` → 3 persons.
  - `"a republic was established under Sun-Yat-Sen (1866-1925)"` → `Sun Yat-sen`.
  - `"In February 1930, Ho Chi Minh (who spent time in France"` → `Ho Chi Minh`.
  - `"Burgula Ramakrishna Rao, Marri Chenna Reddy, J.V. Narsinga Rao and K.V. Ranga Reddy from Telangana"` → 4 persons, including `K.V. Ranga Reddy`.
  - `"hailstorm in Chevella region of Ranga Reddy district"` → no persons.
  - `"Mr. President, Sir, I introduce the Draft"` → no persons.
  - `"Pastor Niemoeller, a resistance fighter"` → `Niemoeller`.
  - `"His parents were Mahalakshimi and Lakshmikanth Rao"` → 2 persons.
  - Output must have exactly 1 row for {Nehru, Jawaharlal Nehru, J. Nehru, Pandit Jawaharlal Nehru}.

## 5. Priority order for round 3
1. **Merge cluster rows into one canonical row** (N10). This is a quick fix that removes about 46 duplicate rows.
2. **Make decisions follow the score, and drop rows below 0.45** (N11). This empties the junk out of review.
3. **Add context filters for rivers, places, calendar words, organisations and title-only spans** (N1–N4). This removes most of the 45 accepted non-names.
4. **Add the dotless-initials pattern and let initials override a place match** (N5, N6).
5. **Extract text column by column and remove figure captions** (N8). Then add the Asian-name and "X of Country" patterns (N7).
6. **Add the missed-name cues** in N12.

Target for round 3, accepted rows on both books: **precision ≥ 90%, recall ≥ 85%, at most 5 duplicates, and less than 30% junk in review.**
