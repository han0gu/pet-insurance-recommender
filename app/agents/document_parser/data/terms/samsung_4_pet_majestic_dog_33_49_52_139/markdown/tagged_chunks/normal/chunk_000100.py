from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다.\n'
 '# 제25조 (보험나이 등)- ① 이 약관에서의 피보험자의 나이는 보험나이를 기준으로 합니다. 다만, 제23조(계약의\n'
 '- 무효) 제2호의 경우에는 실제 만 나이를 적용합니다.\n'
 '- ② 제1항의 보험나이는 계약일 현재 피보험자의 실제 만 나이를 기준으로 6개월 미만의\n'
 '- 끝수는 버리고 6개월 이상의 끝수는 1년으로 하여 계산하며, 이후 매년 계약해당일에\n'
 '- 나이가 증가하는 것으로 합니다.\n'
 '- ③ 피보험자의 나이 또는 성별에 관한 기재사항이 사실과 다른 경우에는 정정된 나이 또'),
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
 'indexing': {'chunk_id': 'chunk_000100',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
