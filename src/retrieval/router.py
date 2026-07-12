from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 


ROUTE_EXAMPLES = {
    "patient_history": [
        "what is the patient history",
        "show me the admission notes",
        "what were the lab results",
        "describe the xray findings"
    ],
    "drug": [
        "what is the dosage of",
        "drug interaction between",
        "contraindications for",
        "side effects of"
    ],
    "trial": [
        "what does the research show",
        "clinical evidence for",
        "trial results"
    ]
}

COLLECTION_MAP = {
    "patient_history": ["patient_records"],
    "drug":            ["patient_records", "pharma_guidelines"],
    "trial":           ["patient_records", "clinical_trials"],
    "general":         ["patient_records", "pharma_guidelines", "clinical_trials"]
}

class SemanticRouter:
    def __init__(self, embedder : HuggingFaceEmbeddings):
        self.embedder = embedder
        self.route_embedder = {
            route : embedder.embed_documents(examples)
            for route, examples in ROUTE_EXAMPLES.items()
        }

    def route(self, query: str):
        query_embedding = self.embedder.embed_query(query)
        scores = {}

        for router, embedding in self.route_embedder.items():
            cosines = cosine_similarity([query_embedding], embedding)[0]
            scores[router] = float(np.max(cosines))
        intent = max(scores, key=scores.get)
        collections = COLLECTION_MAP.get(intent, COLLECTION_MAP["general"])
        return intent, collections