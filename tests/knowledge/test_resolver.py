from __future__ import annotations

from app.knowledge.llm import FakeLLM
from app.knowledge.resolver import EntityResolver
from app.knowledge.schema import (
    EntityMatch,
    EntityType,
    ExtractedEntity,
    ResolvedEntity,
    entity_key,
)


def _known(*entities: ResolvedEntity) -> dict[str, ResolvedEntity]:
    return {entity.key: entity for entity in entities}


PAYMENT = ResolvedEntity(
    key=entity_key("Payment API", EntityType.SERVICE),
    name="Payment API",
    type=EntityType.SERVICE,
)


def test_an_exact_key_match_reuses_the_existing_entity_without_the_model():
    llm = FakeLLM(responses=[])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="payment   API", type=EntityType.SERVICE), _known(PAYMENT)
    )

    assert resolved.key == PAYMENT.key
    assert resolved.name == "Payment API"
    assert llm.prompts == []


def test_punctuation_and_case_differences_resolve_without_the_model():
    llm = FakeLLM(responses=[])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payment-API!", type=EntityType.SERVICE), _known(PAYMENT)
    )
    assert resolved.key == PAYMENT.key
    assert llm.prompts == []


def test_an_unrelated_entity_is_created_new_without_the_model():
    llm = FakeLLM(responses=[])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="PostgreSQL", type=EntityType.DATABASE), _known(PAYMENT)
    )

    assert resolved.key == entity_key("PostgreSQL", EntityType.DATABASE)
    assert llm.prompts == []


def test_the_model_adjudicates_only_a_fuzzy_candidate():
    llm = FakeLLM(responses=[EntityMatch(match=True, matched_key=PAYMENT.key)])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payment API service", type=EntityType.SERVICE),
        _known(PAYMENT),
    )

    assert resolved.key == PAYMENT.key
    assert len(llm.prompts) == 1


def test_a_rejected_fuzzy_match_creates_a_new_entity():
    llm = FakeLLM(responses=[EntityMatch(match=False, matched_key=None)])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payment API v2", type=EntityType.SERVICE), _known(PAYMENT)
    )

    assert resolved.key == entity_key("Payment API v2", EntityType.SERVICE)


def test_a_match_naming_an_unknown_key_is_ignored():
    llm = FakeLLM(responses=[EntityMatch(match=True, matched_key="service|invented")])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payment API gateway", type=EntityType.SERVICE),
        _known(PAYMENT),
    )
    assert resolved.key == entity_key("Payment API gateway", EntityType.SERVICE)


def test_candidates_of_a_different_type_are_not_offered_to_the_model():
    llm = FakeLLM(responses=[])
    other = ResolvedEntity(
        key=entity_key("Payment API", EntityType.DOCUMENT),
        name="Payment API",
        type=EntityType.DOCUMENT,
    )
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payment API docs", type=EntityType.SERVICE), _known(other)
    )

    assert llm.prompts == []
    assert resolved.type is EntityType.SERVICE


def test_an_alias_match_resolves_without_the_model():
    llm = FakeLLM(responses=[])
    aliased = PAYMENT.with_aliases(["payments service"])
    resolved = EntityResolver(llm).resolve(
        ExtractedEntity(name="Payments Service", type=EntityType.SERVICE),
        _known(aliased),
    )

    assert resolved.key == PAYMENT.key
    assert llm.prompts == []


def test_resolution_works_with_no_model_configured():
    resolved = EntityResolver(None).resolve(
        ExtractedEntity(name="Payment API gateway", type=EntityType.SERVICE),
        _known(PAYMENT),
    )
    assert resolved.key == entity_key("Payment API gateway", EntityType.SERVICE)


def test_aliases_from_the_extraction_are_carried_onto_the_resolved_entity():
    resolved = EntityResolver(None).resolve(
        ExtractedEntity(
            name="PostgreSQL", type=EntityType.DATABASE, aliases=["Postgres"]
        ),
        {},
    )
    assert "postgres" in resolved.aliases
