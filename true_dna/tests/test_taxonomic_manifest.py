import build_taxonomic_manifest as taxonomy
import pytest
from build_taxonomic_manifest import load_candidates, validate_selection


def _taxon(tax_id, name, family_id, family_name, domain):
    return {
        "tax_id": tax_id,
        "scientific_name": name,
        "rank": "species",
        "family_tax_id": family_id,
        "family_name": family_name,
        "superkingdom_tax_id": "2",
        "superkingdom_name": domain,
    }


def test_taxonomic_selection_rejects_duplicate_families_before_download():
    candidates = [
        {
            "tax_id": "1",
            "configured_name": "first",
            "domain": "bacteria",
            "weight": 1,
            "selection_reason": "test",
            "selection_priority": 1,
        },
        {
            "tax_id": "2",
            "configured_name": "second",
            "domain": "bacteria",
            "weight": 1,
            "selection_reason": "test",
            "selection_priority": 2,
        },
    ]
    taxonomy = {
        "1": _taxon("1", "first", "100", "Exampleaceae", "Bacteria"),
        "2": _taxon("2", "second", "100", "Exampleaceae", "Bacteria"),
    }

    with pytest.raises(ValueError, match="Duplicate family"):
        validate_selection(candidates, taxonomy)


def test_taxonomic_selection_attaches_ncbi_taxonomy_provenance():
    candidate = {
        "tax_id": "1",
        "configured_name": "first",
        "domain": "bacteria",
        "weight": 1,
        "selection_reason": "test",
        "selection_priority": 1,
    }
    selected = validate_selection([candidate], {"1": _taxon("1", "first", "100", "Exampleaceae", "Bacteria")})

    assert selected[0]["family_name"] == "Exampleaceae"
    assert selected[0]["taxonomy_url"].endswith("id=1")


def test_taxonomic_selection_preserves_audited_refseq_accession(tmp_path):
    config = tmp_path / "selection.json"
    config.write_text(
        '{"species": [{"tax_id": "4932", "scientific_name": "Saccharomyces cerevisiae", '
        '"domain": "eukaryotes", "preferred_refseq_accession": "GCF_000146045.2"}]}',
        encoding="utf-8",
    )

    candidates = load_candidates(config)

    assert candidates[0]["preferred_refseq_accession"] == "GCF_000146045.2"


@pytest.mark.parametrize(
    "operation",
    [
        lambda key: taxonomy._taxonomy_request(["4932"], key, None),
        lambda key: taxonomy._taxonomy_search("Saccharomyces cerevisiae", key, None),
    ],
)
def test_taxonomy_request_failures_never_expose_api_keys(monkeypatch, operation):
    sentinel = "SENTINEL_NCBI_API_KEY_MUST_NOT_LEAK"

    def fail_request(*_args, **_kwargs):
        raise taxonomy.requests.RequestException(f"request failed: https://example.invalid?api_key={sentinel}")

    monkeypatch.setattr(taxonomy.requests, "get", fail_request)
    monkeypatch.setattr(taxonomy.time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError) as exc_info:
        operation(sentinel)

    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
