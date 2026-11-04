"""Closed Rovent corpus wire families; this module performs no conversion.

Only the declared family of container, input and original native bodies is
checked here. The source reader still owns every exact field, byte pin, seal,
target and lineage validation. A matching descriptor confers no execution or
native authority and does not make an input valid for a model.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CorpusCodec:
    """Immutable domain selection, never a parser or an execution capability."""

    manifest_domain: str
    index_domain: str
    record_domain: str
    publication_domain: str
    input_domain: str
    native_domains: tuple[str, ...]

    @property
    def source_domain(self):
        """Keep each original record domain in the lineage projection."""
        return self.record_domain


LEGACY_CODEC = CorpusCodec(
    manifest_domain="rz-rovent-corpus-manifest/1",
    index_domain="rz-rovent-corpus-index/1",
    record_domain="rz-rovent-training-record/1",
    publication_domain="rz-rovent-corpus-publication/1",
    input_domain="rz-pals-generation-input/3",
    native_domains=("rz-rovent-native-owner-event/1", "rz-rovent-native-owner-event/2"),
)
SUBMITTED_CODEC = CorpusCodec(
    manifest_domain="rz-rovent-corpus-manifest/2",
    index_domain="rz-rovent-corpus-index/2",
    record_domain="rz-rovent-training-record/2",
    publication_domain="rz-rovent-corpus-publication/2",
    input_domain="rz-rovent-submitted-input/1",
    native_domains=("rz-rovent-raw-record/3",),
)
_CODECS = (LEGACY_CODEC, SUBMITTED_CODEC)
_RAW_FIELDS = ("raw_submission", "raw_completion", "raw_retirement")


def _object(value, description):
    if type(value) is not dict:
        raise ValueError(description + " must be a plain object")
    return value


def _schema(value, description):
    result = _object(value, description).get("schema")
    if type(result) is not str:
        raise ValueError(description + " requires an explicit string schema")
    return result


def _select(value, attribute, description):
    schema = _schema(value, description)
    for codec in _CODECS:
        if schema == getattr(codec, attribute):
            return codec
    raise ValueError("unsupported " + description + " schema; no automatic conversion")


def codec_from_manifest(value):
    """Select only a registered manifest family; full validation is separate."""
    return _select(value, "manifest_domain", "Rovent corpus manifest")


def codec_from_publication(value):
    """Select only a registered publication family; full validation is separate."""
    return _select(value, "publication_domain", "Rovent corpus publication")


def validate_container_pair(manifest, index):
    """Reject a mixed manifest/index family without changing either body."""
    codec = codec_from_manifest(manifest)
    if _schema(index, "Rovent corpus index") != codec.index_domain:
        raise ValueError("Rovent manifest/index wire family mismatch; no conversion")
    return codec


def validate_record_codec(manifest, record):
    """Check the closed family of one record, not its content or authority.

    The historical Generation V3 input has no ``schema`` field. Its seal
    domain lives outside that body. The submitted input instead carries its
    own mandatory schema. Neither is projected, re-signed or relabelled here.
    """
    codec = codec_from_manifest(manifest)
    if _schema(record, "Rovent training record") != codec.record_domain:
        raise ValueError("Rovent manifest/record wire family mismatch; no conversion")
    snapshot = _object(record.get("input"), "Rovent submitted input")
    if codec is LEGACY_CODEC:
        if "schema" in snapshot:
            raise ValueError("legacy Generation V3 input must remain schema-less; no conversion")
    elif _schema(snapshot, "Rovent submitted input") != codec.input_domain:
        raise ValueError("Rovent record/input wire family mismatch; no conversion")
    native = _object(record.get("native"), "Rovent native record")
    domains = tuple(_schema(native.get(name), "Rovent " + name) for name in _RAW_FIELDS)
    if any(domain not in codec.native_domains for domain in domains):
        raise ValueError("Rovent record/native wire family mismatch; no conversion")
    if len(set(domains)) != 1:
        raise ValueError("Rovent original native bodies must use one schema revision")
    return codec
