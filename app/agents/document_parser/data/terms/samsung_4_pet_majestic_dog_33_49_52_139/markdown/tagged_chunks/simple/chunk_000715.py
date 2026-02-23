from langchain_core.documents import Document

chunk = Document(
    page_content=('| 경과기간 | 기준 | 1년 | 2년 | 3년 | 4년 | 5년 |\n'
 '| 1년미만 | 보험계약에 정한 지급보험금 | 50% | 30% | 25% | 20% | 15% |\n'
 '| 1년이상 2년미만 | 보험계약에 정한 지급보험금 |  | 60% | 50% | 40% | 30% |\n'
 '| 2년이상 3년미만 | 보험계약에 정한 지급보험금 |  |  | 75% | 60% | 45% |\n'
 '| 3년이상 4년미만 | 보험계약에 정한 지급보험금 |  |  |  | 80% | 60% |'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000715',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
