from langchain_core.documents import Document

chunk = Document(
    page_content=('- 64 -\n'
 '※ 약관에서 인용된 법·규정은「별표 및 참고」의 「약관에서 인용된 법·규정」에서 확인할 수 있습니다.\n'
 '1. 상해 관련 특별약관\n'
 '제1관 일반사항'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 68},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000342',
              'chunk_char_len': 86,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
