from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 제1호, 제2호 및 제3호에도 불구하고 [갱신형] 특별약관 중 해약환급금 구분이 해 약환급금 미지급형 및 해약환급금 미지급형Ⅱ을 '
 '제외한 경우에는 해당 특별약관의 보험기간 중 계약이 해지될 경우 “보험료 및 해약환급금 산출방법서”에 따라 계산 한 금액을 해약환급금으로 '
 '지급합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 64},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000320',
              'chunk_char_len': 158,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
