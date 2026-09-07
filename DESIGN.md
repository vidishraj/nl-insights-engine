# Design — NL Insights Engine

A service that answers natural-language questions about **any transactional CSV it has
never seen**, inferring the schema from the file itself, with no code changes per
dataset.

---

## 1. The thesis

Most "ask your CSV" systems wire a language model straight to SQL generation. That
fails in a specific, dangerous way: the SQL is *syntactically valid and references real
columns*, yet is **wrong about meaning** — it sums a per-unit price, ranks postage as a
product, or reports gross where the user meant net. The system answers confidently and
you cannot tell it was wrong without already knowing the answer.

This design refuses that failure mode by making one move:

> **The system's understanding of a file is a first-class, machine-checked artifact — a
> Semantic Model with evidence and confidence — and every answer is a typed plan
> validated against that model before any SQL runs. The LLM proposes; deterministic
> code disposes.**

The load-bearing invariant, enforced structurally rather than by prompt discipline:

> **There is no code path from a question to executed SQL that does not pass the
> binder.**

The interpreter emits a *typed plan* (never SQL, never free text); the binder validates
every node of that plan against the inferred model and returns exactly one verdict —
`ANSWERABLE` / `ANSWER_WITH_CAVEATS` / `REFUSE` / `CLARIFY` — carrying evidence. Only a
bound plan reaches the executor. An invented column or an unresolvable measure cannot
become SQL; it becomes a refusal with a reason.

A second consequence follows from treating proposal and disposal as separate stages: the
**LLM is not load-bearing**. Role proposal has two interchangeable sources — the model,
or a deterministic heuristic over the same profiler evidence — and both are tested by
the *same* verifiers. "Heuristics propose, code disposes" is the identical architecture
to "the model proposes, code disposes," so the system **degrades instead of dying**
without a key, and still refuses to invent.

![architecture](docs/architecture.svg)

*Ingest → propose → verify → freeze the Semantic Model; then interpret → bind → execute,
with the binder as the one gate no question bypasses.*

---

## 2. Components — what each owns

| Package | Owns | Artifact it produces |
| --- | --- | --- |
| `ingestion` | dialect sniffing, full-file type inference, evidence-based date-order resolution, the deterministic **profile** | `Profile` (types, null rates, cardinalities, sign mix, integer-dominance, functional dependencies) |
| `semantic` | role **proposal** (LLM *or* heuristic) and deterministic **disposal** — verifiers, returns/partition/label/period discovery, the measure algebra | `SemanticModel` (bindings + evidence + confidence + provenance) |
| `interpreter` | natural language → a typed plan (structured output; never SQL); follow-ups as IR merge | `QueryIR` |
| `binder` | the **refusal spine** — validates the plan against the model, injects correctness defaults | `Verdict` (one of four kinds, evidence-carrying) |
| `executor` | compile a bound plan to SQL, run it, render a **plan-derived** explanation | `Answer` (rows + SQL + formula + caveats + coverage) |
| `provider` | the LLM seam: one interface, three auth paths (replay / apikey / ambient), record + replay | `LLMProvider` |
| `jobs` | async orchestration — submit / stream (SSE) / cancel, atomic promotion, per-dataset serialisation, idempotency | `Job`, `Dataset` registry |
| `api` | thin HTTP adapter over `jobs`; shaped errors only | FastAPI app |
| `eval` | golden verdict suites + refusal confusion matrix, metamorphic ablation, header-stripping, anti-hardcoding lint | reports + CI gates |

The boundaries are chosen on purpose. Two examples worth calling out:

- **The binder is DB-free; the executor holds the connection.** The binder *decides*
  (does this plan bind? which correctness defaults apply?) using only the model. When a
  decision needs data — e.g. quantifying how much postage a non-partitioned revenue
  total actually contains — that work lives in the executor, which has the DuckDB
  connection. The disclosure is decided in the binder and *quantified* in the executor.
- **Exploration happens at ingest time, not question time.** The expensive
  understanding of a file (profiling, FD discovery, verification) is done once, when the
  file arrives, and frozen into the Semantic Model. A question is then a cheap, typed
  lookup against that artifact — not an agent re-exploring the data per query.

---

## 3. The path a question takes

```
CSV ─▶ ingestion ─▶ Profile ─▶ semantic ─▶ SemanticModel
                                                 │
question ─▶ interpreter ─▶ QueryIR ─▶ binder ────┤─▶ REFUSE / CLARIFY   (no SQL runs)
                                                 │
                                          bound plan ─▶ executor ─▶ Answer
```

1. **Ingestion** loads the CSV into a per-dataset DuckDB file and computes the Profile
   by SQL over the whole table (never a sample). Dialect is sniffed; a ragged file fails
   loudly; `dd/mm` vs `mm/dd` is settled by evidence (a day component > 12), not locale.
2. **Semantic build** proposes a role for every column (LLM or heuristic), then *tests
   every testable claim in SQL*: does this key share a timestamp across its rows? does
   `amount == quantity × rate` hold row-wise? It discovers the returns convention, the
   fact partition, entity labels, and period completeness — from evidence, never from
   names — and resolves the measure algebra. A refuted claim is demoted and unbound.
3. **Interpretation** turns the question into a `QueryIR` via forced structured output —
   measures by name, dimensions by column, filters, time windows, top-k. It never writes
   SQL. Follow-ups produce a *partial* IR that is merged onto the previous plan (a diff),
   so the model never replays chat history.
4. **Binding** validates the IR against the model and returns one verdict; its
   deterministic refusal detectors each carry evidence. On the answerable
   path it injects correctness defaults (net-of-returns assumption, product-line
   partition, incomplete-period exclusion, low-coverage caveat).
5. **Execution** compiles the bound plan to SQL, runs it, and renders the explanation
   **from the plan** — so the explanation cannot lie about what was computed. The Answer
   carries the SQL, the formula over roles, the filters applied (including binder
   defaults), coverage, and every caveat.

---

## 4. How schema context is built from an unfamiliar file — and where it breaks

### How it is built

The Semantic Model is assembled in stages, each of which emits a progress event so a UI
can watch it form (`sniffing → loading → profiling → resolving_dates → inferring →
verifying → discovering → promoting`):

1. **Profile (pure SQL).** Types, null rates, cardinality, uniqueness, min/max, numeric
   sign mix, integer-dominance, top values, samples, and **functional dependencies with
   strength *and* support** — the last computed only over multi-row groups, because a
   one-row group trivially "determines" every column and would inflate strength.
2. **Propose roles.** Either the LLM (given the evidence pack as structured input, told a
   column name is one weak signal among the stats) or the heuristic proposer (rules over
   the same evidence). Each claim carries a confidence and a **provenance** (`llm` /
   `heuristic`).
3. **Verify (pure SQL).** Every claim implying a testable property is tested: a
   transaction key's rows share one event_time; a stored amount equals quantity × rate;
   an entity key's coverage is recorded as a fact (e.g. 25% null), not an error. Only
   *strong* disproofs refute — a structural check informs but never overrides a valid
   claim (a whole-number price `$5.00` is still a rate).
4. **Discover conventions.** Returns (explicit flag / explicit category / key-prefix /
   negative-quantity), the fact partition (a dimension that determines a flag splitting
   product from non-product rows), entity labels (a description FD-linked to an entity
   key), and period completeness — all from FDs and correlations, never names.
5. **Resolve the measure algebra** over roles: `net_revenue`, `gross_revenue`,
   `units_sold`, `order_count`, `basket_cooccurrence` — available only if their roles are
   bound and non-refuted.

The same file therefore yields different, *correct* models depending on the evidence it
carries. "Top products by revenue" on the synthetic slice **excludes** postage (a
`line_type` column carries the evidence, so the partition fires); on raw UCI it
**includes** postage (no such column exists, so there is nothing to act on). The system
acts on evidence, not names, and declines to invent structure a file does not have.

### Where it breaks — the honest taxonomy

This is the part I want read closely. Some of these the system **catches and refuses**;
one it **cannot catch in principle**, and saying so is worth more than pretending
completeness.

| Failure class | What goes wrong | Status |
| --- | --- | --- |
| **Pre-aggregated rows** | If a file is already rolled up (one row per product-month with a summed quantity), a `quantity × rate` revenue **double-counts** or misprices. The stored-amount verifier catches the *presence* of a total, but cannot know rows are pre-aggregated. | **Partially caught** — prefers a verified stored amount when present; otherwise a known risk, disclosed below. |
| **Multiple money columns** | `price` / `discount` / `tax` / `total` are ambiguous: which is *the* amount? Picking wrong silently changes every revenue answer. | **Mitigated** — the stored-amount identity (`amount ≈ qty × rate`) disambiguates the extended amount; when several amounts compete it **clarifies**, listing the candidates rather than picking one, and when a single amount is present but unverified it is **summed as reported with an explicit caveat** naming the column and that the identity could not be confirmed — neither a silent guess nor a blanket refusal. |
| **Pivoted data (months as columns)** | `Jan, Feb, Mar…` as columns is not a transactional table at all; every row-wise assumption is void. | **Should refuse at ingestion** (a named "this looks pivoted, not transactional" refusal) — see the cut list. |
| **Snapshot vs event data** | If rows are balances/snapshots rather than events, summing is meaningless. | **Caught indirectly** — the additive-quantity and transaction-key verifiers fail (no shared-time key, no additive structure), so measures don't bind. |
| **Multi-currency without a currency column** | Amounts in mixed currencies with **no column recording which** are indistinguishable from single-currency data. Summing them is wrong, and **nothing in a single file can reveal it**. | **Cannot be caught, in principle.** The system will confidently sum them. This is the one class of error the architecture *cannot* detect, and the honest boundary of "understand any file." |
| **Thin files + `min_support=5`** | FD strength is computed only over multi-row groups with ≥5 of them. A 200-order file with only 4 multi-line orders **loses** `order_id → order_date`, so the transaction key may not verify and `order_count` may not bind. | **Known suppression** — deliberate (singletons are unfalsifiable), but it means genuine dependencies on thin files are dropped. A confidence-scaled `min_support` is future work. |
| **A concept the file lacks** | Asked for a concept a file does not carry (`profit`, an unknown grouping like "by supplier region"), a naive system substitutes the nearest available measure and hides the swap. | **Caught** — the interpreter records what it could not express as `unmet_concepts`, and the binder **refuses**, naming the concept it could not bind rather than answering a different question (`ir.py` → `interpret.py` → `bind.py`, pinned by `test_binder.py`). The deterministic path already refused an unbound *measure*; this closes the interpreter's eager-substitution gap too. |

Two design responses make these honest rather than silent: measures **refuse** when
their roles don't bind (better a refusal than a wrong number), and answerable-but-scoped
results **disclose and quantify** their assumptions and caveats (net-of-returns; and exactly how much
non-product money a bare total contains). Against the committed synthetic slice the bare `net_revenue` disclosure reads, verbatim from the code:

> includes 6 non-PRODUCT line_type value(s) totalling +59,256.00 (POSTAGE +60,000.00, FEE -480.00, CARRIAGE +360.00, +3 more); restrict to line_type = PRODUCT to exclude them

The definition behind the figure, stated because the number cannot imply it: it sums the **bound revenue measure** over rows whose `line_type` *string* is not the discovered fact value (`PRODUCT`), under the query's own filters; contributors are ordered by absolute value and the **top three are named with a `+N more` tail** while the total covers all of them. The count is of NON-ZERO non-fact groups - a group whose partition value is null, or whose measure is null or exactly zero, is dropped - so on another file it can read fewer values than a reader counting the raw column sees, and be correct. On this fixture the flag, the string, and `quantity × unit_price` all agree, so every reasonable route reproduces the total — which is why the figure alone cannot reveal which basis produced it.

### The bug class we hunt: two components disagreeing about what a value *means*

Adversarial review found the same shape of defect repeatedly, and it is worth naming
because it is the class a validated pipeline is uniquely able to catch — and uniquely able
to *hide* if the check is sloppy. Each is a boundary where a claim about the data went
unchecked against what the data, or another component, actually did: usually two layers
agreeing on a value's *bytes* but not its *meaning*, sometimes one component announcing work
another never performed. They are listed in discovery order; each names its fix:

- **Presence, not type.** The binder checked that an `event_time` role was *bound*, but not
  that the column was actually a timestamp — a VARCHAR date compiled to SQL that returned a
  silent NULL. *Fix: verify the type, not just the presence.*
- **A probe weaker than the system's own knowledge.** The temporal check probed a text date
  with `TRY_CAST` (0%) while the ingester had already resolved the format (`strptime`, 100%)
  — a verifier that asks a weaker question than the system can answer refutes true claims.
  *Fix: probe with the instrument you already have.*
- **A contract in a comment.** `TimeWindow.end` was documented exclusive in Python, but the
  schema handed to the model never said so, so the model could emit an inclusive bound.
  *Fix: put the contract on the surface the other component actually reads (the schema).*
- **Names from one namespace checked against another.** The interpreter named a thing by its
  *role* (`transaction_key`) or *entity kind* (`customer`); the binder checked those against
  *column* names, which never match, and refused with a false reason. *Fix: resolve across
  namespaces at the boundary; pick one contract and make both sides honour it.*
- **Substitution instead of refusal.** Asked for `profit` (no cost data), the model answered
  `revenue` and presented it as the answer. *Fix: a structured "I could not express this"
  channel (`unmet_concepts`) so the binder refuses honestly rather than inventing.*
- **Absence read as emptiness (`dict.get` conflating the two).** The post-execution
  degeneracy check judged the IR's `group_by` (`StockCode`), but the basket path rewrites its
  projection to `product_a`/`product_b` — so `row.get("StockCode")` returned `None` for an
  *absent key* exactly as for a *null value*, attaching a false "the grouping carries no
  signal" caveat to a correct answer. This is the inverse of a silent wrong answer: a right
  answer discredited by a wrong disclaimer, which corrodes the caveat machinery precisely
  where it must be believed. *Fix: judge the columns the **result** carries, not the plan's —
  require the key present before calling its values null; plan projection and result schema
  are different namespaces (the same disease as the role/column case above).*
- **A hard cast that assumed a type the data did not promise.** The returns/partition
  verifiers cast a two-valued flag to `INTEGER` to test it — but a flag can be *text*
  (`'charge'`/`'refund'`), so `CAST('charge' AS INTEGER)` threw and failed the whole ingest of
  an unseen file. One component assumed "flag ⇒ numeric"; the data said otherwise. *Fix:
  `TRY_CAST` (a non-numeric flag becomes NULL, not a crash) and treat a text two-valued marker
  as a category. Discovered because an adversarial file carried two bugs in series — a timeout,
  then this crash behind it — the case for testing against files you have never seen.*
- **A name checked, a value invented.** Asked "how many returns were there," the model
  filtered `invoice_type IN ('return')` on a column whose only values are
  `SALE`/`CANCELLATION`/`ADJUSTMENT`. The binder validated the column *name* but never the
  *value*, so it executed to a confident, uncaveated `0` — and the degeneracy guard missed it
  because an ungrouped aggregate over zero rows returns *one row of zeros*, not an empty set,
  while the guard only checked for `if not rows`. *Fix: validate filter values against the
  column's observed set, but only refuse (CLARIFY) when that set is **exhaustive** — a sample
  proves nothing about absence, so a value missing from a sampled column is left to run; and
  teach the degeneracy guard the second shape of "nothing matched" (a zero-match aggregate,
  detected by a cheap EXISTS over the same filters) so a zero is never mistaken for an answer.*
- **A component that reports on work another component never did.** Asked for `net revenue of
  the past 2 years`, the binder emitted the caveat "time window: last 2 year" and the UI echoed
  "filtered where time on invoice_date" — but the executor's `_time()` had no `last_n` branch,
  so it compiled *no* WHERE clause and returned the whole-table total. The number was right only
  because that dataset spanned under a year; on any longer file it is silently wrong *with an
  affirmative false claim of a filter attached*. This is the sharpest form of the class: not an
  omission but a positive false statement, one subsystem announcing a restriction another never
  applied. *Fix: compile `last_n` (a real WHERE, anchored on the data's latest date and that
  anchor disclosed), and a structural guard — a "window" caveat now asserts the compiler
  produced a matching time clause, so the announcement and the work cannot diverge again.*
- **A requested *shape* silently dropped, disclosed only where nobody looks.** Asked for
  `revenue by region` on a file with no region column, the interpreter dropped the grouping,
  returned one ungrouped scalar, and recorded the drop in `ir.notes` — the interpreter's
  internal commentary, not the `caveats`/`assumptions` surface the UI and API render. A
  breakdown had silently become a total: the answer's *shape* changed into the answer to a
  different question, and the one honest sentence about it sat in the least visible field.
  *Fix: a structured `unmet_dimensions` channel (mirroring `unmet_concepts`) — an unavailable
  grouping/filter is surfaced, not dropped, and the binder turns it into a CLARIFY that names
  the gap and offers the dimensions that do exist. Disclosure must live on the surface the
  consumer actually reads, and a dropped request must change the verdict, never just a note.*
- **A contract with a future version of ourselves, broken by not bumping the version.** The
  persistence layer correctly refuses an *unknown* model version — but a new field the binder
  relies on (`categorical_values`) was added *without* bumping the version, so a sidecar written
  before the fix loaded as "compatible" while missing the data, and the dataset ran in a silent
  degraded mode (the filter-value CLARIFY never fired for datasets ingested earlier). The two
  components that disagreed were the current binder and a sidecar written by an *older version of
  ourselves*. *Fix: bump the version when the schema gains a binder-relied-upon field, and treat an
  older-but-known version as "must re-derive" — recompute the missing field from the still-on-disk
  DuckDB file (no LLM, no loss) and upgrade the sidecar, rather than refusing or degrading. A
  version field only helps if it is actually bumped when the contract changes.*

- **Numbers computed beside the plan, desynchronised from it.** A later review of the brief's
  five named questions found four whose *arithmetic* had drifted from the plan the binder
  validated. `share_of_total` summed the **post-LIMIT** rows, so a "top 5" denominator was the
  five, not the population, and every share was ~5–19× too large while summing to exactly 1.0 —
  a wrongness shaped like a passing check. The `frequency` denominator grouped by an entity with
  **no NULL guard**, so the anonymous pseudo-entity's revenue sat in the denominator *while the
  caveat named the opposite scope* — the same false-disclosure class as the `last_n` case. The
  `basket` count was `count(*)` over a self-join (line-pairs, not baskets), inflating every
  number and silently counting returns. `period_comparison` never computed growth at all: it
  returned the per-period rows unranked and left the delta "to the reader," a comment describing
  a consumer that did not exist. And filter `op` was a free-form `str`, so an unrecognised
  operator (`'!='`, `'not in'`) hit `dict.get(op, default)` and selected the **opposite** branch
  — "revenue excluding cancelled orders" returning exactly the cancelled orders. *Fixes: compute
  the share denominator in SQL over the un-limited result; exclude NULL entities and derive the
  caveat from the same predicate; `count(DISTINCT transaction)` with the product partition and
  returns excluded, disclosed; actually pivot the two periods, rank by the delta, and CLARIFY a
  period that is not in the data; make the operator a `Literal` (a schema `enum` the model reads)
  the binder re-checks, and index the compile maps directly so an unknown op raises rather than
  defaults. The incomplete-period exclusion now carries the same structural guard as the window
  caveat — a "period" caveat asserts the clause is in the SQL, so the disclosure and the work
  cannot diverge.* The through-line of all five: **post-plan arithmetic is still part of the
  answer, and it must be validated against the plan, not trusted because the plan was.**

A coda on the degeneracy fix: the *first* wording of that fix's comment named the sample
dataset's grouping column, and the anti-hardcoding lint — the check that makes "nothing about
the sample dataset is hardcoded" falsifiable rather than merely asserted — went red on it. The
guardrail caught its own author sneaking a dataset name into the source. That is the whole
thesis in miniature: the machine-checked artifact holds even when the human writing it forgets.

A coda on *verifying* the fixes, which is a lesson in its own right: the review's sharpest
finding was that the original pre-submission check claimed all five named questions passed — and
it was wrong about three, because its ground truth was **built from the same assumption as the
thing it checked**. A basket ground truth written with the same `count(*)`-over-self-join shape
as the code would agree with the code and both would be wrong. So the regression tests for these
five compute their ground truth from the data's *meaning* — distinct invoices, identified
customers, the whole-population denominator, a two-period delta — with SQL written **without
reference to the engine's aggregation shape**, and assert the engine equals it through the real
`execute`/`run_query` call site. Independent verification has to be independent in its
*assumptions*, not merely run separately; and because the buggy and fixed engines differ on the
committed synthetic data, each test fails if its fix is reverted.

The through-line: **a claim must be validated up front, against the meaning the consumer
will assume, not left for DuckDB to reject at run time or for a reader to catch.** The
typed-IR-plus-binder architecture is what makes these findable at all — each was a boundary
with a name and a test — but only if the check interrogates *meaning*, not just shape.

---

## 5. The biggest decisions (and the alternatives rejected)

**1. A typed IR + binder, not text-to-SQL.**
Rejected: prompt the LLM to emit SQL over the real columns. That SQL can be *valid and
still wrong about meaning* — the failure this whole system exists to prevent. An IR node
either binds to a verified role or it does not; "wrongness about meaning" becomes a
structural bind failure, not a plausible-looking query. The cost is expressiveness (the
IR covers a deliberate menu of question shapes, not arbitrary SQL); the benefit is that
every answer is checkable by construction.

**2. Refusal as a structural subsystem, not a prompt instruction.**
Rejected: "if you're not sure, say you can't answer." Models comply with that
inconsistently and unfalsifiably. Here refusal is *code*: deterministic detectors,
each carrying evidence, on a path no question can bypass. It is testable: the binder
mutation check in the eval flips a green suite red when a detector is disabled.

**3. DuckDB, one file per dataset — over SQLite / Postgres / Polars.**
DuckDB gives full-file columnar analytics in-process with zero server, `read_csv` with
whole-file type inference, and an Excel/`xlsx` path. One file *per dataset* is also the
concurrency answer: writers for different datasets never contend, so "how does this
behave under load" is *isolation by construction* rather than a lock to reason about.
Postgres would add an operational dependency for no analytical gain; SQLite lacks the
columnar/analytic ergonomics; Polars would put the engine in-process memory rather than a
queryable, inspectable store.

**4. Exploration at ingest time, frozen into an artifact — not an agent exploring per
question.**
Rejected: a per-question agent that pokes at the data to figure out what a column means.
That is slow, non-deterministic, and re-derives the same understanding on every query.
Doing the expensive understanding once, at ingest, and freezing it into a Semantic Model
makes questions cheap, answers reproducible, and — crucially — makes the understanding
**inspectable** before any question is asked. Making it **correctable** — a human overriding
a binding the system got wrong — is the first thing I would add (section 6); it is deliberately
not built yet, because a corrected binding must be re-verified and provenance-tracked rather
than trusted on assertion, which is a design question rather than an endpoint.

**5. The LLM proposes, but is not required (heuristic fallback).**
Rejected: making the semantic layer depend on an LLM call. The credential-free replay
path could then not ingest an *unseen* file — the one thing the product is for. Instead a
deterministic heuristic proposes roles from the same evidence and the same verifiers test
it. The LLM improves proposal *quality*; it is not load-bearing for the system to work.
Sharpened: **the heuristic binds only what the data corroborates; where a NAME is the only
disambiguator, it abstains and the system refuses.** A per-unit rate and a stored amount
are statistically identical — both decimal — so on a file with several money columns (a
discount, a price, a shipping fee) picking "the first decimal" as the rate is a coin flip
presented as a fact, and it computes revenue from the discount with no caveat. The
heuristic instead commits a money role only when the choice is forced (a single decimal) or
the stored-amount identity (`amount == quantity × rate`) confirms the triple downstream;
otherwise it leaves rate/amount unbound and a revenue question REFUSES. Telling a price from
a discount is a naming question a statistics-only proposer *cannot* answer, so declining is
the correct answer, not a limitation — it is the same thesis as the binder's refusals.

This difference is **measured**, not assumed (both proposers run over the same evidence;
the diff is recorded in `fixtures/llm/`). On the raw UCI file the heuristic — with no
column-name understanding and a text-typed date column — cannot find the transaction key
or the event time, so it labels `InvoiceNo`, `InvoiceDate`, `StockCode`, `Description` all
as generic `entity_key/other` and loses two whole measures (`order_count`,
`basket_cooccurrence`). The live model types all eight columns correctly
(`InvoiceNo`→transaction_key 0.98, `InvoiceDate`→event_time 0.99, `StockCode`→product,
`CustomerID`→customer, `Description`→description, `Country`→dimension) and both measures
come back. On the demo file the gap is smaller but the same shape: the heuristic marks
`product`/`customer` as generic dimensions (0.4), the model types them as entities (~0.9)
and unlocks basket analysis. So the LLM is a real quality lift on proposal — and the
system still *runs and refuses correctly* without it. That is the thesis, with numbers.

---

## 6. What I'd build next, in order

1. **Human correction of the semantic model.** The model is inspectable but not yet
   editable: there is no endpoint to override a binding the system got wrong. Building it
   means an endpoint that accepts a corrected binding, RE-VERIFICATION of the edited model
   rather than trusting the edit, and a provenance record distinguishing a human-asserted
   binding from a machine-verified one. That last point is why it is not an afternoon: a
   corrected model must not silently inherit the confidence of a verified one, and deciding
   how it should carry its confidence is a design question, not an endpoint.
2. **Pivoted-input refusal at ingestion** — detect wide/pivoted files (many
   same-typed numeric columns whose headers are periods) and refuse with a named reason,
   rather than mis-modelling them.
3. **Confidence-scaled `min_support`** — let thin files keep genuine dependencies by
   weighting strength by support instead of hard-gating at 5.
4. **Recorded golden fixtures for the interpreter** — commit replay fixtures for the
   example questions so the *full* NL→answer path (not just the deterministic core) runs
   green in CI.
5. **Currency-column heuristic + explicit unknown-currency caveat** — where a currency
   column *does* exist, group by it; where it does not, attach an explicit "assumed
   single-currency" caveat so the one uncatchable class is at least *named* in the answer.
6. **Bound the evidence pack by column count.** The role-proposal prompt embeds per-column
   statistics, samples, and top-values, so its size — and the LLM latency — scales with the
   number of *columns*, a dimension the user controls and we do not cap. A 10-column file
   already needs minutes; a 100-column CSV would build an enormous prompt. The live timeout
   is now generous and env-tunable, which stops a hang, but the honest fix is to cap
   samples/top-values per column (and total columns) as width grows, so latency degrades
   gracefully instead of relying on a big timeout.
7. **App-level dataset reclamation.** Ingested datasets persist (DuckDB + a JSON sidecar
   rehydrated on startup, verified for identity against the data before it is trusted), but
   nothing in the app reclaims them — an open, no-auth endpoint accretes datasets until the
   disk guard starts declining uploads. Deployment covers it with a server-side prune timer;
   the app itself should own a TTL / LRU eviction so capacity management is not purely
   operational. Related: rehydration currently collapses four distinct refusal causes
   (unknown version, unreadable/rebuilt data, missing file, malformed sidecar) into one
   "not loaded" outcome — each is logged, but a surfaced status would aid an operator.
8. **Dataset removal through the API.** Datasets can be created but not removed: clearing one
   today means deleting its sidecar and DuckDB file on the server and restarting. A `DELETE`
   endpoint is deliberately *not* built on this pass, because its lifecycle questions deserve a
   considered answer rather than a late one — what happens to an in-flight job against the
   dataset, whether it is idempotent, whether it should require confirmation — and rushing those
   on the last change before submission is the wrong trade in a system whose whole pitch is
   refusing rather than guessing. This is distinct from the reclamation above: that is automatic
   eviction under capacity pressure; this is an explicit, caller-driven removal.
9. **Measure labels derived from the columns, not fixed retail nouns.** There are two axes of
   overfitting, and only one is machine-checked. The anti-hardcoding lint proves no sample-*file*
   token is baked in; but the measure NAMES — `net_revenue`, `units_sold`, `order_count` — are
   sample-*domain* vocabulary the lint cannot see, so a freight or payroll file that binds every
   role correctly still had to be asked about "revenue". The current fix mitigates this at the
   interpreter: each measure is shown with the concrete columns it aggregates, so the model maps
   a domain term ("total freight cost") to the measure over that column. The deeper fix is to
   *derive* each measure's label from its bound columns rather than a fixed noun, so the system
   speaks the dataset's language in its answers too — it touches the measure algebra, the binder,
   and the goldens, so it is named here rather than rushed in.

### Trusting the LLM where the deterministic layer over-vetoed — two problems, not one

A later review against unfamiliar open files (a supermarket export, restaurant tips, diamonds,
gapminder, a passenger manifest) found the system *refusing* questions the data plainly
supported. The instructive framing is that this was **two** distinct defects, and only one is
"trust the LLM more":

- **A verifier bug — a confirming test used as a disqualifying one.** The stored-amount identity
  (`amount ≈ quantity × rate`) is *sufficient* evidence a column is an extended amount, but it was
  used as *necessary*: a total carrying tax (`total = quantity × rate × 1.05`) FAILED it and was
  refuted to confidence 0, so a **cost** column that happened to satisfy the identity became
  "revenue". Refutation must require positive contrary evidence — the rule the quantity/rate
  structure verifiers already followed ("a weak signal, never a disproof"); the amount identity
  was the lone violator. *Fix: a failed confirming test marks UNVERIFIED, not REFUTED; the amount
  is selected per-column so a "verified" label is only ever earned by the column actually summed;
  a single unverified amount is summed AS REPORTED with a prose disclosure that names the column
  and says the check could not be **run** (distinct from failing it); and when **more than one**
  column is proposed as the amount, the binder CLARIFYs and lists them rather than silently
  picking the one that satisfies the identity (the cost).*
- **A capability gap — the measure algebra was retail-only.** `net_revenue`/`units_sold`/
  `order_count` did not cover "how many rows", "average of a numeric", "max of a value the
  proposer discarded". This is **not** distrust: the model's numeric proposal was accepted, there
  was simply no measure that consumed it. *Fix: a generic aggregation (`count(*)`, and
  `avg/sum/min/max` over any column numeric by stored type, regardless of role) — the executor
  already runs a bound measure, so it is near-free downstream.* This is also why an explicit
  `sum(numeric_id)` now answers rather than refuses: the role layer governs what the system
  *infers* about a column, but an aggregation the user *names* is an instruction, honoured as
  given rather than second-guessed — unlike the appendix's "id a naive heuristic would sum as a
  rate", which was a role the system *chose* for a column nobody asked about, the opposite case.

None of this loosened the refusal discipline; it corrected *where* the discipline applied. The
honest line is between **a choice made without evidence** and **no choice to make**: one amount
column is nothing to resolve, so summing it with a disclosure that the identity could not be run
is the only available reading, not a guess; several amount columns are a real choice, so the
system clarifies and lists them. The old behaviour had this exactly backwards — it refused where
nothing had to be decided and guessed where something did.

**Deliberately NOT built: a smarter amount disambiguator.** When several money columns compete,
one could *infer* the revenue column (e.g. prefer the additive superset that the others sum into).
That is new inference logic, in the exact subsystem that had just produced a silent wrong number,
and it trades a cheap CLARIFY for a confident guess. The judgment here is to **ask, not guess** —
the CLARIFY is the correct terminal behavior, and the inference is out of scope on purpose.

---

## 7. The cut list — deliberate engineering judgment

Shipped less than the full vision on purpose; here is what was cut and why, so the
trade-offs are legible.

- **Recorded fixtures cover the demo end-to-end, and role proposal on both files.** The
  bundled demo ships **real** cassettes recorded from a live model — role proposal *and*
  five interpretations — so the zero-credential quickstart runs the whole "LLM proposes,
  code disposes" pipeline hermetically (a test pins them against key drift). Role proposal
  is recorded for raw UCI too (that is where the LLM-vs-heuristic diff is largest). What is
  cut is a *full* recorded question set against the two large files; the automated suites
  stand in there by driving the deterministic core (bind → execute) at the IR level, where
  the graded properties — refusal, correctness, generality — all live. **Cut: full
  recorded interpreter fixtures for the large files.**
- **Pivoted-input detection is documented, not implemented.** It is a clean named
  refusal and I know exactly where it goes (ingestion), but it was lower value than the
  correctness traps on the two real files. **Cut: the pivoted refusal detector.**
- **The UI is last and minimal.** Per the brief's priority, the analytical core, the
  service, and the eval come first; the conversational UI is a thin SSE client over an
  API that is already complete. **Cut: UI polish.**
- **Real-fixture metamorphic ablation and headers-stripped suites are hermetic, not run
  against the 500k-row files in CI.** The mechanism is proven on synthetic data every
  run; running it against the large fixtures is disk-heavy and adds coverage, not
  confidence. **Cut: large-fixture eval runs in CI.**

None of these cut a *correctness* guarantee — they cut breadth of coverage and polish.
The invariant (no SQL without the binder), the refusal spine, the disclosure discipline,
and the no-credential path are all shipped and tested.

---

## Appendix — evidence the guarantees are real

- **The anti-hardcoding lint caught its own author.** It found five development-dataset
  column names in the engine's *own comments* and failed the build until they were
  genericised. A rule that has already caught its author is more convincing than one
  asserted to be strict.
- **The binder fails closed, and the eval proves the checks are load-bearing.** The
  eval's binder-mutation check replaces the binder's verdict with a blanket "answerable"
  and the golden suite catches it — the false-answer rate jumps above zero and the suite
  goes red — so a green suite means the refusal checks are load-bearing, not decorative.
  And a *stronger* property shows up when you disable one detector in place rather than
  the whole verdict: bypassing the missing-concept check does **not** yield a plausible
  wrong answer — the binder cannot build a plan for a measure the model never bound and
  raises (`KeyError: 'profit'`) instead. No-plan-at-all beats a metric ticking up; the
  system fails toward silence, not toward a confident invention.
- **Multi-agent development surfaced the traps.** The correctness defects that most
  matter — a partition keyed to the group-by column instead of the ranked entity, a
  one-directional disclosure, a role a verifier disproved but the binder still used, an
  id column a naive heuristic would sum as a rate — were each found by an adversarial
  reviewer driving the real fixtures and the real HTTP API, then fixed with a regression
  test that pins the ground truth.
