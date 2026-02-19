from langchain_core.documents import Document

chunk = Document(
    page_content=('. 5) "발가락뼈 일부를 잃었을 때" 라 함은 첫째 발가락의 지관절, 다른 네 발가락 의 제1지관절(근위지관절)부터 심장에서 먼 쪽으로 '
 '발가락 뼈 일부가 절단된 경우를 말하며, 뼈 단면이 불규칙해진 상태나 발가락 길이의 단축 없이 골편만 떨어진 상태는 해당하지 않는다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 146},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000960',
              'chunk_char_len': 150,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
