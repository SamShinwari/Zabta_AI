from __future__ import annotations

from typing import Any, Callable


class FBRGenerator:
    """
    LLM generation layer for the FBR QA system.

    The generator receives grounded FBR context from the QA
    layer and sends a final prompt to an LLM backend.

    The backend can be injected as a callable, which makes
    the class easy to test without requiring Ollama.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        backend: Callable[[str], str] | None = None,
    ):
        self.model = model
        self.backend = backend

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        prompt_or_question: str,
        qa_result: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate an answer.

        Two modes are supported:

        1. Direct prompt mode:

            generate(prompt)

        2. Pipeline mode:

            generate(question, qa_result)

        In pipeline mode, qa_result should contain a
        prepared/context/prompt representation.
        """

        self._validate_prompt_or_question(
            prompt_or_question
        )

        if qa_result is None:
            prompt = prompt_or_question

        else:
            if not isinstance(
                qa_result,
                dict,
            ):
                raise TypeError(
                    "qa_result must be a dictionary"
                )

            prompt = self._build_pipeline_prompt(
                question=prompt_or_question,
                qa_result=qa_result,
            )

        return self._call_backend(
            prompt
        )

    # ========================================================
    # GENERATE FROM PREPARED
    # ========================================================

    def generate_from_prepared(
        self,
        prepared: dict[str, Any],
    ) -> str:
        """
        Generate an answer from FBRQA.prepare().
        """

        if not isinstance(
            prepared,
            dict,
        ):
            raise TypeError(
                "prepared must be a dictionary"
            )

        if "prompt" not in prepared:
            raise ValueError(
                "prepared result does not contain 'prompt'"
            )

        return self.generate(
            prepared["prompt"]
        )

    # ========================================================
    # PIPELINE PROMPT
    # ========================================================

    def _build_pipeline_prompt(
        self,
        question: str,
        qa_result: dict[str, Any],
    ) -> str:
        """
        Convert the QA result into a grounded generation
        prompt.
        """

        # ----------------------------------------------------
        # Prefer an already-built prompt.
        # ----------------------------------------------------

        existing_prompt = qa_result.get(
            "prompt"
        )

        if isinstance(
            existing_prompt,
            str,
        ) and existing_prompt.strip():

            return existing_prompt

        # ----------------------------------------------------
        # Extract context.
        # ----------------------------------------------------

        context = qa_result.get(
            "context",
            "",
        )

        if not isinstance(
            context,
            str,
        ):
            context = str(context)

        # ----------------------------------------------------
        # Extract evidence answer if available.
        # ----------------------------------------------------

        evidence_answer = qa_result.get(
            "answer",
            "",
        )

        if not isinstance(
            evidence_answer,
            str,
        ):
            evidence_answer = str(
                evidence_answer
            )

        # ----------------------------------------------------
        # Build grounded prompt.
        # ----------------------------------------------------

        return (
            "You are an FBR sales tax compliance "
            "assistant.\n"
            "\n"
            "Answer the user's question using ONLY "
            "the provided FBR evidence.\n"
            "\n"
            "Rules:\n"
            "1. Do not invent legal requirements.\n"
            "2. Do not use information outside the evidence.\n"
            "3. If the evidence is insufficient, clearly "
            "say so.\n"
            "4. Prefer the most recent applicable source.\n"
            "5. Give a concise and clear answer.\n"
            "6. Cite relevant source numbers when available.\n"
            "\n"
            f"USER QUESTION:\n{question}\n"
            "\n"
            f"FBR EVIDENCE:\n{context}\n"
            "\n"
            f"EVIDENCE-BASED ANSWER:\n{evidence_answer}\n"
            "\n"
            "FINAL ANSWER:"
        )

    # ========================================================
    # BACKEND
    # ========================================================

    def _call_backend(
        self,
        prompt: str,
    ) -> str:
        """
        Call the configured LLM backend.
        """

        self._validate_prompt(
            prompt
        )

        if self.backend is None:
            raise RuntimeError(
                "No LLM backend configured. "
                "Provide a backend callable."
            )

        answer = self.backend(
            prompt
        )

        if not isinstance(
            answer,
            str,
        ):
            raise TypeError(
                "LLM backend must return a string"
            )

        answer = answer.strip()

        if not answer:
            raise RuntimeError(
                "LLM backend returned an empty answer"
            )

        return answer

    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_prompt_or_question(
        value: str,
    ) -> None:

        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                "prompt_or_question must be a string"
            )

        if not value.strip():
            raise ValueError(
                "prompt_or_question cannot be empty"
            )

    @staticmethod
    def _validate_prompt(
        prompt: str,
    ) -> None:

        if not isinstance(
            prompt,
            str,
        ):
            raise TypeError(
                "prompt must be a string"
            )

        if not prompt.strip():
            raise ValueError(
                "prompt cannot be empty"
            )