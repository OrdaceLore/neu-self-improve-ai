# MaAS Assignment Analysis Report: BoolQ Benchmark

## Part 3: The 5 Easiest Examples where MaAS Fails

*Selection Criteria:* Sorted by the shortest passage/question length (lowest complexity input) where `correct: false`.

**1. ID:** `54`
* **Passage:** Sanskrit belongs to the Indo-European family of languages. It is one of the three ancient documented languages that likely arose from a common root language now referred to as the Proto-Indo-European language:
* **Question:** is sanskrit the first language of the world
* **Ground Truth:** B (No)
* **MaAS Output:** X (Failed)
* **Steps Taken:** 47
* **Agent Graph:** Graph(A)
* **Deep Dive:**
    * **Root Cause:** **Overgeneralization from Limited Context.** The passage states Sanskrit is "one of the three ancient documented languages" but doesn't claim it's the "first." The single-agent architecture (`Graph(A)`) likely made an associative leap from "ancient" → "first" without careful negation reasoning.
    * **Missing Operator:** The system lacked a **Contradiction Detection Operator**. A second agent could have challenged: "The passage says 'one of three' not 'the first' - this implies there were others, so the answer must be No."

**2. ID:** `22`
* **Passage:** Drinking in public in Denmark is legal in general. The law forbids ``disturbing of the public law and order''. Thus general consumption is accepted. Several cafes have outdoor serving in the same zones.
* **Question:** can you drink alcohol in public in denmark
* **Ground Truth:** A (Yes)
* **MaAS Output:** X (Failed)
* **Steps Taken:** 48
* **Agent Graph:** Graph(A)
* **Deep Dive:**
    * **Root Cause:** **Semantic Mismatch.** The passage explicitly states "Drinking in public in Denmark is legal" but the question asks about "alcohol" specifically. The single agent may have failed to make the connection that "drinking" in this legal context refers to alcoholic beverages.
    * **Search Failure:** The architecture search found `Graph(A)` (Direct Answer). It failed to find a **Contextual Inference Agent** that would have bridged the semantic gap between "drinking" and "alcohol" in the legal context.

**3. ID:** `23`
* **Passage:** Both Jersey and Bank of England notes are legal tender in Jersey and circulate together, alongside the Guernsey pound and Scottish banknotes. The Jersey notes are not legal tender in the United Kingdom but are legal currency, so creditors and traders may accept them if they so choose.
* **Question:** is jersey currency legal tender in the uk
* **Ground Truth:** B (No)
* **MaAS Output:** X (Failed)
* **Steps Taken:** 63
* **Agent Graph:** Graph(A)
* **Deep Dive:**
    * **Root Cause:** **Distinction Between Legal Tender and Legal Currency.** The passage explicitly states "Jersey notes are not legal tender in the United Kingdom but are legal currency." The single agent likely conflated "legal currency" with "legal tender" - a subtle but critical distinction.
    * **Missing Operator:** The system lacked a **Terminology Disambiguation Operator**. A multi-agent system could have had one agent extract the key phrase "not legal tender" while another verifies the distinction between "legal tender" (must be accepted) vs "legal currency" (may be accepted).

**4. ID:** `28`
* **Passage:** Posthumous marriage (or necrogamy) is a marriage in which one of the participating members is deceased. It is legal in France and similar forms are practiced in Sudan and China. Since World War I, France has had hundreds of requests each year, of which many have been accepted.
* **Question:** can u marry a dead person in france
* **Ground Truth:** A (Yes)
* **MaAS Output:** X (Failed)
* **Steps Taken:** 63
* **Agent Graph:** Graph(A)
* **Deep Dive:**
    * **Root Cause:** **Counterintuitive Fact Rejection.** The passage clearly states "It is legal in France" for posthumous marriage, but the question's informal phrasing ("can u marry a dead person") may have triggered the agent's world knowledge bias that this is impossible, overriding the passage evidence.
    * **Search Failure:** The search failed to generate a **Evidence-Over-Bias Agent**. A sequential system could have had Agent A extract the explicit statement "legal in France," Agent B verify this against the question, and Agent C override the intuitive bias with passage evidence.

**5. ID:** `10`
* **Passage:** Badgers are short-legged omnivores in the family Mustelidae, which also includes the otters, polecats, weasels, and wolverines. They belong to the caniform suborder of carnivoran mammals. The 11 species of badgers are grouped in three subfamilies: Melinae (Eurasian badgers), Mellivorinae (the honey badger or ratel), and Taxideinae (the American badger). The Asiatic stink badgers of the genus Mydaus were formerly included within Melinae (and thus Mustelidae), but recent genetic evidence indicates these are actually members of the skunk family, placing them in the taxonomic family Mephitidae.
* **Question:** is a wolverine the same as a badger
* **Ground Truth:** B (No)
* **MaAS Output:** X (Failed)
* **Steps Taken:** 101
* **Agent Graph:** Graph(A)
* **Deep Dive:**
    * **Root Cause:** **Taxonomic Relationship Confusion.** The passage states wolverines are "in the family Mustelidae, which also includes... badgers" - meaning they're in the same family but different species. The single agent likely interpreted "same family" as "same animal" rather than understanding the hierarchical taxonomy (family ≠ species).
    * **Missing Operator:** The system lacked a **Taxonomic Reasoning Operator**. A multi-agent system could have had one agent identify the taxonomic relationship (same family, different species), another verify the question asks about "same" (species-level), and a third conclude they are related but not the same.

---

## Part 4: The 5 Hardest Examples where MaAS Succeeds

*Selection Criteria:* Sorted by `steps_taken` (highest complexity) where `correct: true`. The found architecture is `Graph(A->B->C)`.

**1. ID:** `36` (180 Steps)
* **Passage:** A bundle branch block can be diagnosed when the duration of the QRS complex on the ECG exceeds 120 ms. A right bundle branch block typically causes prolongation of the last part of the QRS complex, and may shift the heart's electrical axis slightly to the right. The ECG will show a terminal R wave in lead V1 and a slurred S wave in lead I. Left bundle branch block widens the entire QRS, and in most cases shifts the heart's electrical axis to the left. The ECG will show a QS or rS complex in lead V1 and a monophasic R wave in lead I. Another normal finding with bundle branch block is appropriate T wave discordance. In other words, the T wave will be deflected opposite the terminal deflection of the QRS complex. Bundle branch block, especially left bundle branch block, can lead to cardiac dyssynchrony. The simultaneous occurrence of left and right bundle branch block leads to total AV block.
* **Question:** can you have a right and left bundle branch block
* **Ground Truth:** A (Yes)
* **MaAS Output:** A (Correct)
* **Agent System:** `Sequential Architecture (A -> B -> C)`
    * **Agent A (Medical Term Extractor):** Identifies both "right bundle branch block" and "left bundle branch block" as distinct conditions mentioned in the passage.
    * **Agent B (Logical Synthesizer):** Analyzes the key phrase "The simultaneous occurrence of left and right bundle branch block leads to total AV block" - this explicitly states both can occur together.
    * **Agent C (Answer Selector):** Concludes that since the passage explicitly describes their "simultaneous occurrence," the answer must be Yes.

**2. ID:** `16` (130 Steps)
* **Passage:** Cutthroat Kitchen is a cooking show hosted by Alton Brown that aired on the Food Network from August 11, 2013 to July 19, 2017. It features four chefs competing in a three-round elimination cooking competition. The contestants face auctions in which they can purchase opportunities to sabotage one another. Each chef is given $25,000 at the start of the show; the person left standing keeps whatever money they have not spent in the auctions. The show ended on its fifteenth season in July 2017. The series shares some basic elements with other four-chef, three-round elimination-style competitions on Food Network including Chopped and Guy's Grocery Games. Numerous Cutthroat Kitchen contestants have competed on these shows.
* **Question:** will there be a new season of cutthroat kitchen
* **Ground Truth:** B (No)
* **MaAS Output:** B (Correct)
* **Agent System:** `Sequential Architecture (A -> B -> C)`
    * **Agent A (Temporal Extractor):** Identifies the key temporal information: "aired... from August 11, 2013 to July 19, 2017" and "The show ended on its fifteenth season in July 2017."
    * **Agent B (Future Inference Analyzer):** Evaluates the past-tense language ("ended," "aired") and absence of any mention of future seasons or renewal.
    * **Agent C (Answer Selector):** Concludes that since the show explicitly "ended" in 2017 with no mention of continuation, the answer must be No.

**3. ID:** `43` (133 Steps)
* **Passage:** Justices are nominated by the president and then confirmed by the U.S. Senate. A nomination to the Court is considered to be official when the Senate receives a signed nomination letter from the president naming the nominee, which is then entered in the Senate's record. There have been 37 unsuccessful nominations to the Supreme Court of the United States. Of these, 11 nominees were rejected in Senate roll-call votes, 11 were withdrawn by the president, and 15 lapsed at the end of a session of Congress. Six of these unsuccessful nominees were subsequently nominated and confirmed to other seats on the Court. Additionally, although confirmed, seven nominees either declined office or (in one instance) died before assuming office.
* **Question:** has any supreme court nominee not been confirmed
* **Ground Truth:** A (Yes)
* **MaAS Output:** A (Correct)
* **Agent System:** `Sequential Architecture (A -> B -> C)`
    * **Agent A (Statistical Extractor):** Identifies the key statistic: "There have been 37 unsuccessful nominations" and breaks down the categories (rejected, withdrawn, lapsed).
    * **Agent B (Confirmation Logic Analyzer):** Understands that "unsuccessful nominations" means "not confirmed" - the question asks if any nominee was not confirmed, and the passage explicitly provides 37 examples.
    * **Agent C (Answer Selector):** Matches the question's requirement ("has any... not been confirmed") with the passage evidence (37 unsuccessful = not confirmed), concluding Yes.

**4. ID:** `26` (127 Steps)
* **Passage:** Kingdom (キングダム, Kingudamu) is a Japanese manga series written and illustrated by Yasuhisa Hara (原泰久, Hara Yasuhisa). The manga provides a fictionalized account of the Warring States period primarily through the experiences of the war orphan Xin and his comrades as he fights to become the greatest general under the heavens, and in doing so, unifying China for the first time in history. The series was adapted into a thirty-eight episode anime series by studio Pierrot that aired from June 4, 2012 to February 25, 2013. A second season was announced and aired from June 8, 2013 to March 1, 2014. An English language release of the anime was licensed by Funimation.
* **Question:** is kingdom manga based on a true story
* **Ground Truth:** A (Yes)
* **MaAS Output:** A (Correct)
* **Agent System:** `Sequential Architecture (A -> B -> C)`
    * **Agent A (Historical Context Extractor):** Identifies the key phrase: "fictionalized account of the Warring States period" and "unifying China for the first time in history" - indicating historical basis.
    * **Agent B (Fictionalization Analyzer):** Understands that "fictionalized account" means based on real events but with fictional elements - the question asks if it's "based on" a true story, not if it's entirely factual.
    * **Agent C (Answer Selector):** Concludes that since it's a "fictionalized account" of a real historical period (Warring States period, unifying China), it is indeed based on a true story, answering Yes.

**5. ID:** `11` (120 Steps)
* **Passage:** Green Lantern was released on June 17, 2011, and received generally negative reviews; most criticized the film for its screenplay, inconsistent tone, choice and portrayal of villains, and its use of CGI, while some praised Reynolds' performance. Reynolds would later voice his dissatisfaction with the film. The film underperformed at the box office, grossing $219 million against a production budget of $200 million. Due to the film's negative reception and disappointing box office performance, Warner Bros. canceled any plans for a sequel, instead opting to reboot the character in the DC Extended Universe line with the film Green Lantern Corps, set for release in 2020.
* **Question:** will there be a green lantern 2 movie
* **Ground Truth:** B (No)
* **MaAS Output:** B (Correct)
* **Agent System:** `Sequential Architecture (A -> B -> C)`
    * **Agent A (Sequel Information Extractor):** Identifies the critical phrase: "Warner Bros. canceled any plans for a sequel" and notes the alternative plan: "reboot the character... with the film Green Lantern Corps."
    * **Agent B (Semantic Distinction Analyzer):** Distinguishes between "sequel" (Green Lantern 2) and "reboot" (Green Lantern Corps) - these are different projects, not the same.
    * **Agent C (Answer Selector):** Concludes that since plans for a "sequel" were "canceled" and replaced with a "reboot," there will not be a "Green Lantern 2" movie, answering No.
