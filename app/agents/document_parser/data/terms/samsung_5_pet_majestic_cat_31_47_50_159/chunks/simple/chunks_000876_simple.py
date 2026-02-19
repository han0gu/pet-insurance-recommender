from langchain_core.documents import Document

chunk = Document(
    page_content=('. 9) "눈꺼풀에 뚜렷한 결손을 남긴 때" 라 함은 눈꺼풀의 결손으로 눈을 감았을 때 각막(검은자위)이 완전히 덮이지 않는 경우를 '
 '말한다. 10) "눈꺼풀에 뚜렷한 운동장해를 남긴 때" 라 함은 눈을 떴을 때 동공을 1/2 이 상 덮거나 또는 눈을 감았을 때 각막을 '
 '완전히 덮을 수 없는 경우를 말한다. 11) 외상이나 화상 등으로 안구의 적출이 불가피한 경우에는 외모의 추상(추한 모 습)이 가산된다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 137},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['eye', 'skin']},
 'indexing': {'chunk_id': 'chunk_000876',
              'chunk_char_len': 225,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
