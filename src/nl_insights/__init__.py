"""NL Insights Engine — answer natural-language questions about any transactional CSV.

Architecture (each package owns one artifact; arrows are typed data, not calls):

    CSV -> ingestion -> Profile -> semantic -> SemanticModel
    question -> interpreter -> QueryIR -> binder -> (plan | refuse | clarify)
             -> executor -> Answer
    jobs wraps ingestion+query (ids, status, streaming); api exposes it; eval grades it.

The spine invariant: no path from question to executed SQL bypasses the binder.
"""

__version__ = "0.1.0"
