import logging
from typing import List, Optional, Dict, Any
from app.config import (
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    GEMINI_MODEL,
    OPENAI_MODEL,
    LLM_TIMEOUT_SECONDS,
    PERSONA_JUNIOR_ANALYST,
    PERSONA_PORTFOLIO_MANAGER,
    PERSONA_COMPLIANCE_REVIEWER,
)
from app.models import DocumentChunk
from app.generation.personas import PersonaGuidelines

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM Client providing real LLM generation (Google Gemini / OpenAI)
    with clean fallback handling to deterministic rule-based synthesis.
    """

    @classmethod
    def is_configured(cls) -> bool:
        """Checks if at least one LLM provider API key is configured."""
        return bool(GEMINI_API_KEY or OPENAI_API_KEY)

    @classmethod
    def get_active_provider(cls) -> Optional[str]:
        if GEMINI_API_KEY:
            return f"Gemini ({GEMINI_MODEL})"
        elif OPENAI_API_KEY:
            return f"OpenAI ({OPENAI_MODEL})"
        return None

    @classmethod
    def build_prompt(
        cls,
        persona: str,
        query: str,
        query_type: str,
        chunks: List[DocumentChunk],
    ) -> Dict[str, str]:
        """Constructs system and user prompts calibrated for the requested persona."""
        guidelines = PersonaGuidelines.get_persona_instructions(persona)
        persona_name = guidelines["name"]

        # Format context chunks
        context_blocks = []
        for c in chunks:
            ref = c.chunk_id.replace("_", " ").title()
            block = (
                f"[{ref}] (Document: {c.metadata.document_name}, "
                f"Page: {c.metadata.page_number}, Type: {c.metadata.document_type.upper()}):\n"
                f"{c.content.strip()}"
            )
            context_blocks.append(block)

        context_str = "\n\n".join(context_blocks)

        rules_str = "\n".join([f"- {r}" for r in guidelines.get("rules", [])])

        system_prompt = (
            f"You are a Principal Financial Research Assistant operating strictly as a {persona_name}.\n"
            f"Persona Tone & Role: {guidelines['tone']}\n\n"
            f"CORE PERSONA GUIDELINES:\n{rules_str}\n\n"
            "STRICT GROUNDING & CITATION RULES:\n"
            "1. You must formulate your response using ONLY facts, numbers, and statements directly present in the RETRIEVED CONTEXT below.\n"
            "2. Do NOT hallucinate, extrapolate, or invent figures, dates, or financial ratios not supported by the context.\n"
            "3. Cite source evidence using chunk markers like [Chunk 1], [Chunk 2], etc.\n"
            "4. Required structural sections based on your persona:\n"
        )

        if persona_name == PERSONA_PORTFOLIO_MANAGER:
            system_prompt += (
                "- Header: '### Portfolio Manager Brief: <Title>'\n"
                "- Persona perspective line: '**Persona Perspective**: Portfolio Manager — Executive Summary, Valuation Multiples, and Balance Sheet Risk.'\n"
                "- Section 1: '#### 1. Executive Investment Verdict & Key Takeaways'\n"
                "- Section 2: '#### 2. Valuation & Risk-Adjusted Multiples' (discuss forward P/E, valuation metrics)\n"
                "- Section 3: '#### 3. Balance Sheet Solvency & Liquidity Constraints' (discuss debt-to-equity, current ratio, interest coverage, actionable recommendation)\n"
            )
        elif persona_name == PERSONA_COMPLIANCE_REVIEWER:
            system_prompt += (
                "- Header: '### Compliance & Audit Review: <Title>'\n"
                "- Persona perspective line: '**Persona Perspective**: Compliance Reviewer — Regulatory Audit Trail, Reconciliations, and Data Limitations.'\n"
                "- Section 1: '#### 1. Evidentiary Audit Trail & Chunk Mapping' (list source nodes, document names, pages, and key excerpt text)\n"
                "- Section 2: '#### 2. Reconciled Financial Disclosures'\n"
                "- Section 3: '#### 3. Material Risk Disclosures & Limitations' (disclose interest rate sensitivity, forward-looking guidance risks, and unaudited interim scope caveats)\n"
            )
        else:  # Junior Analyst
            system_prompt += (
                "- Header: '### Financial Research Analysis: <Title>'\n"
                "- Persona perspective line: '**Persona Perspective**: Junior Analyst — Step-by-Step Financial Reasoning & Accounting Deconstruction.'\n"
                "- Section 1: '#### 1. Core Financial Findings & Direct Evidence' (list source chunks with document name, page, and excerpt)\n"
                "- Section 2: '#### 2. Step-by-Step Financial Concepts & Accounting Mechanics' (step-by-step breakdown with definitions of gross margin, COGS, EPS flow-through, and debt mechanics)\n"
                "- Section 3: '#### 3. Management Remediation Actions'\n"
                "- Section 4: '#### 4. Source Chunk Citations' (structured list of cited chunks)\n"
            )

        user_prompt = (
            f"RESEARCH QUERY: {query}\n"
            f"QUERY INTENT: {query_type}\n\n"
            f"RETRIEVED CONTEXT CHUNKS:\n{context_str}\n\n"
            f"Synthesize an authoritative, highly structured financial research assessment strictly following your persona rules and chunk citations."
        )

        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    @classmethod
    def generate_persona_response(
        cls,
        persona: str,
        query: str,
        query_type: str,
        chunks: List[DocumentChunk],
    ) -> Optional[str]:
        """
        Executes real LLM generation using the configured provider.
        Returns generated string if successful, or None on failure/missing key to trigger fallback.
        """
        if not cls.is_configured():
            logger.debug("No LLM API keys configured. Using deterministic rule-based synthesis.")
            return None

        prompts = cls.build_prompt(persona, query, query_type, chunks)
        system_prompt = prompts["system_prompt"]
        user_prompt = prompts["user_prompt"]

        # 1. Primary Provider: Google Gemini
        if GEMINI_API_KEY:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(
                    api_key=GEMINI_API_KEY,
                    http_options=types.HttpOptions(timeout=LLM_TIMEOUT_SECONDS * 1000),
                )
                full_contents = f"{system_prompt}\n\n---\n\n{user_prompt}"

                config = types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.95,
                )

                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=full_contents,
                    config=config,
                )
                if response and response.text:
                    logger.info("Successfully generated persona response via Google Gemini (%s)", GEMINI_MODEL)
                    return response.text.strip()
            except Exception as e:
                logger.warning(
                    "Google Gemini generation failed: %s. Attempting fallback.",
                    str(e),
                )

        # 2. Secondary Provider: OpenAI
        if OPENAI_API_KEY:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS)
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                if response.choices and response.choices[0].message.content:
                    logger.info("Successfully generated persona response via OpenAI (%s)", OPENAI_MODEL)
                    return response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(
                    "OpenAI generation failed: %s. Attempting fallback.",
                    str(e),
                )

        logger.info("LLM generation unavailable or failed; proceeding with deterministic rule-based synthesis.")
        return None
