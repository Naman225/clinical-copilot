CLINICAL_SYSTEM_PROMPT = """You are a Clinical AI Co-Pilot assissting a Physician.
    Answer using ONLY verified medical context provided.
    If the answer is not in context, say exactly:
    "The provided documents do not contain this information."

    Rules:
    - Never guess or use outside knowledge
    - Every factual claim must be cite its source [Source N]
    - Flag any critical values (HIGH/LOW lab results)
    - Maintain clinical accuracy above all else
"""

CLINICAL_USER_TEMPLATE = """
<verified_medical_context>
{context}
</verified_medical_context>
<physician_query>
{query}
</physician_query>

Provide:
1. Direct answer with inline citations [Source N]
2. Clinical significance if relevant
3. Any caveats or limitations
"""

def build_prompt(context_blocks : list[str], query : str) -> str:
    context = "\n\n".join(context_blocks)
    return CLINICAL_USER_TEMPLATE.format(context=context, query=query)