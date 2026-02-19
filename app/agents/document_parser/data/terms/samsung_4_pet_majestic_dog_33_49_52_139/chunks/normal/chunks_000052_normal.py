from langchain_core.documents import Document

chunk = Document(
    page_content=('<예시안내>\n'
 '[보험금을 나누어 지급받을 경우]\n'
 '보험금: 6천만원, 보험금 지급일자: 2024년 4월 1일 일때 보험금을 일시에 지급받지 않고 3년간 매 년 동일한 금액으로 나누어 '
 '지급받는 경우\n'
 '지급일 | 지급액\n'
 '2024년 4월 1일 | 2천만원\n'
 '2025년 4월 1일 | 2천만원 × (1 + 평균공시이율)\n'
 '2026년 4월 1일 | 2천만원 × (1 + 평균공시이율)2\n'
 '[보험금을 일시에 지급받을 경우]'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 37},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000052',
              'chunk_char_len': 224,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
