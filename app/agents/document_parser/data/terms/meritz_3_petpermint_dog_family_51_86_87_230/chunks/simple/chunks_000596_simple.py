from langchain_core.documents import Document

chunk = Document(
    page_content=('제7조(보험금의 분담)\n'
 '\uf000 회사는 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합니다)이 있을 경우 각 계약에 대하여 '
 '다른 계약이 없는 것으로 하여 각각 산출 한 보상책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 손해를 보상합니다. 이 특별약관과 '
 '다른 계약이 모두 의무보험인 경우에도 같습니다.\n'
 '이 특별약관의 보상책임액\n'
 '손해액 × 다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 178},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000596',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
