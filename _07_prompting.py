from typing import List, Dict

def build_prompt(query: str, context_str: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are an expert agricultural engineer specializing in citrus and orange diseases "
        "in Egypt, particularly in the Beheira Governorate. Your answers must be based solely "
        "on the provided context. When you use information from the context, you MUST cite your "
        "sources explicitly by mentioning the source number, file name, and page number "
        "(e.g., [Source 1] File: citrus_diseases.pdf, Page: 12). "
        "If the context does not contain the answer, say that you don't have enough information "
        "and do not make up an answer. Always answer in the same language as the question "
        "(Arabic or English). Provide practical, actionable advice when possible, including "
        "authorized pesticides and treatment methods if mentioned in the context."
    )
    user_prompt = f"Question: {query}\n\nRelevant Context:\n{context_str}\n\nPlease provide a detailed, well-structured answer with source citations."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    return messages
