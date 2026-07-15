from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 


ROUTE_EXAMPLES = {
    "patient_history": [
        "what is the patient history",
        "show me the admission notes",
        "what were the lab results",
        "describe the xray findings",
        "give me a clinical summary of this patient",
        "what is the diagnosis",
        "what are the vital signs",
        "patient medications and treatment plan",
        "show the ECG findings",
        "blood pressure and heart rate",
        "what imaging was done",
        "summarize this patient case",
        "tell me about this patient",
        "what is the patient presenting with",
        "any abnormal lab values",
        "chest x-ray report",
    ],
    "drug": [
        "what is the dosage of",
        "drug interaction between",
        "contraindications for",
        "side effects of",
        "prescribing information for",
        "what are the warnings for this medication",
    ],
    "trial": [
        "what does the research study show",
        "clinical trial evidence for treatment",
        "trial results and outcomes",
        "randomized controlled trial findings",
        "research paper on treatment efficacy",
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