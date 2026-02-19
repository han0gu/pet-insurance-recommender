from langchain_core.documents import Document

chunk = Document(
    page_content='. 4) 코의 추상(추한 모습)장해를 수반한 때에는 기능장해의 지급률과 추상장해의 지 급률을 합산한다.',
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 138},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head', 'other']},
 'indexing': {'chunk_id': 'chunk_000889',
              'chunk_char_len': 57,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
