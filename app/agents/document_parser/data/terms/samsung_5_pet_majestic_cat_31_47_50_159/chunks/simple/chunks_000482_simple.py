from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나, 피보험자가 사망하여 상기 검사 방법을 진단의 기초로 할 수 없는 경우에 한하여 피보험자가 특정법정감염병으로 진 단 또는 '
 '치료를 받고 있었음을 증명할 수 있는 문서화된 기록 또는 증거를 진단확정 의 기초로 할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000482',
              'chunk_char_len': 130,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
