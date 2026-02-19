from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 장해분류표에 해당되지 않는 후유장해는 피보험자의 직업, 연령, 신분 또는 성별 등에 관계없이 신체의 장해정도에 따라 장해분류표의 '
 '구분에 준하여 지급액을 결정합니다. 다만, 장해분류표의 각 장해분류별 최저 지급률 장해정도에 이르지 않는 후유장해에'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 34},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['head', 'joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000014',
              'chunk_char_len': 140,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
