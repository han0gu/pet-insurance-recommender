from langchain_core.documents import Document

chunk = Document(
    page_content=('을 해지할 수 있습니다.- 1. 계약자 또는 피보험자가 고의 또는 중대한 과실로 제11조(계약 전 알릴 의무)를 위\n'
 '- 반하고 그 의무가 중요한 사항에 해당하는 경우\n'
 '- 2. 뚜렷한 위험의 증가와 관련된 제12조(계약 후 알릴 의무) 제1항에서 정한 계약 후\n'
 '- 알릴 의무를 계약자 또는 피보험자의 고의 또는 중대한 과실로 이행하지 않았을\n'
 '- 때\n'
 '# ② 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 이 특별'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000499',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
