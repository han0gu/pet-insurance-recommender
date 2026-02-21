from langchain_core.documents import Document

chunk = Document(
    page_content=('년 동일한 금액으로 나누어 지급받는 경우| 지급일 | 지급액 |\n'
 '| --- | --- |\n'
 '| 2024년 4월 1일 | 2천만원 |\n'
 '| 2025년 4월 1일 | 2천만원 × (1 + 평균공시이율) |\n'
 '| 2026년 4월 1일 | 2천만원 × (1 + 평균공시이율)2 |\n'
 '- 52 -[보험금을 일시에 지급받을 경우]\n'
 '보험금: 3년간 매년 2천만원씩 지급, 보험금 지급 개시일자: 2024년 4월 1일 일때 보험금을 나누어'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000176',
              'chunk_char_len': 233,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
