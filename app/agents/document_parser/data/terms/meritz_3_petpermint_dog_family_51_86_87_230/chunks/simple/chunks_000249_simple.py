from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 계약자는 제4항에 따른 재가입안내와 재가입여부 확인 요청을 받은 경우 재가입 의사를 표시하여야 합니다. \uf000 제4항 '
 '및 제5항에도 불구하고, 회사가 계약자의 재가입 의사를 확인하지 못한 경우(계약자와의 연락두절로 회사의 안내가 계약자에게 도달하지 못한 '
 '경우 포함)에는 직전계약 과 동일한 조건으로 보험계약을 연장합니다. 다만, 보험계 약이 연장된 경우 연장된 날 기준으로 매년 현재의 '
 '예정기'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000249',
              'chunk_char_len': 219,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
