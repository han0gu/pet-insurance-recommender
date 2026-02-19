from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자 또는 피보험자가 고의 또는 중대한 과실로 제16조(계약 전 알릴 의무)를 위 반하고 그 의무가 중요한 사항에 해당하는 경우 '
 '2. 뚜렷한 위험의 증가와 관련된 제17조(상해보험계약 후 알릴 의무) 제1항에서 정한 계약 후 알릴 의무를 계약자 또는 피보험자의 고의 '
 '또는 중대한 과실로 이행하지 않았을 때\n'
 '② 제1항 제1호의 경우에도 불구하고 다음 중 하나에 해당하는 경우에는 회사는 계약을'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 39},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000074',
              'chunk_char_len': 222,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
