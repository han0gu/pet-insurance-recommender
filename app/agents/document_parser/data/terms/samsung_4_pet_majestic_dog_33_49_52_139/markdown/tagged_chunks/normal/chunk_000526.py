from langchain_core.documents import Document

chunk = Document(
    page_content=('- 별로 회사가 정하는 이율로 계산한 금액을 더하여 납입하여야 합니다. 다만, 금리연동\n'
 '- 형보험은 각 상품별 사업방법서에서 별도로 정한 이율로 계산합니다.\n'
 '- ② 제1항에 따라 해지된 특별약관을 부활(효력회복)하는 경우에는 제11조(계약 전 알릴\n'
 '- 의무), 제13조(알릴 의무 위반의 효과), 제15조(사기에 의한 계약), 제20조(제1회 보험\n'
 '- 료 및 회사의 보장개시) 및 보통약관 제20조(보험계약의 성립)를 준용합니다.\n'
 '- ③ 제1항에서 정한 특별약관의 부활(효력회복)이 이루어진 경우라도 계약자 또는 피보험'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000526',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
