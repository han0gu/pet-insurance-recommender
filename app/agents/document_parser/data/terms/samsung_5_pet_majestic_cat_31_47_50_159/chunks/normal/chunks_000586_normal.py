from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 이 계약을 청약할 때 반려묘의 건강상태를 판단할 수 있는 기초자료(건강진 단서 사본 등)에 따라 승낙한 경우에 건강진단서 사본 '
 '등에 명기되어 있는 사항으'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000586',
              'chunk_char_len': 91,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
