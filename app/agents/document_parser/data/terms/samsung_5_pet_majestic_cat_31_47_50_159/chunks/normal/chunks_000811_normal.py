from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[할증위험률에 의한 보험료] 피보험자의 건강상태가 회사가 정한 기준에 적합하지 않은 경우 일반위험률보다 높은 위험률을 적 용하여 산출된 '
 '보험료를 말합니다. [표준체 보험료] 할증위험률의 가입조건(보험기간, 납입기간, 피보험자의 가입나이 등)과 동일한 기준에서, 일반위\n'
 '험률을 적용하여 산출된 보험료를 말합니다.\n'
 '2. 보험금감액법'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 127},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000811',
              'chunk_char_len': 190,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
