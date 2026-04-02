import uuid
from typing import List
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """
    Representation of a snippet's origin, noting position mapping.
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    Object handling fundamental queries without resolved contexts.
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    Extends UnansweredQuestion adding exact source mapping and resolved texts.
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    General dataset wrapping multi-QA pairs.
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    Struct specifying retrieved documents related to a certain inquiry.
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    Answer mapping a set of documents back to a concrete question response.
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    Top 'k' aggregation metric for source retrievals per queries limit testing.
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """
    An interface handling query answers with evaluation pairs per sources.
    """
    search_results: List[MinimalAnswer]
