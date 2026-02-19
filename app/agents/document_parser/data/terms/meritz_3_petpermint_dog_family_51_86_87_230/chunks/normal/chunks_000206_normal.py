from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조(지급보험금의 계산)\n'
 '\uf000 동일한 반려동물과 동일한 사고에 관하여 보험금을 지급 하는 다른 계약(공제계약을 포함합니다)이 있을 경우 각 계 약에 대하여 '
 '다른 계약이 없는 것으로 하여 각각 산출한 지 급보험금의 합계액이 피보험자가 부담한 비용금액을 초과할 때에는 아래에 따라 보험금을 '
 '지급합니다.\n'
 '피보험자가 이 계약의 지급보험금 부담한 총 × 다른 계약이 없는 것으로 하여 각각 계산한 비용금액 지급보험금의 합계액'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 94},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000206',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
