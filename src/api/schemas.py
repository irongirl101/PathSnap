"""Stable JSON contracts shared by the PathSnap API and browser UI."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GPSPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    timestamp: datetime | None = None


class MapMatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trip_id: str | None = Field(default=None, max_length=128)
    points: list[GPSPoint] = Field(min_length=2, max_length=100_000)


class MatchedPoint(BaseModel):
    input_index: int = Field(ge=0)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    node_id: str | int | None = None
    edge_id: str | int | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class MapMatchResponse(BaseModel):
    trip_id: str | None = None
    route_arc_ids: list[str | int]
    points: list[MatchedPoint] = Field(default_factory=list)
    distance_m: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    trip_ids: list[str] | None = None


class EvaluationResponse(BaseModel):
    evaluated_trips: int = Field(ge=0)
    metrics: dict[str, float | int | None]
    per_trip: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
