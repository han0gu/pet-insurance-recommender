from langchain_core.documents import Document

chunk = Document(
    page_content=('제15조(특별약관의 무효)\n'
 '이 특별약관을 가입할 때에 보험사고가 이미 발생하였을 경 우 이 특별약관은 무효로 합니다. 다만, 회사의 고의 또는 과실로 특별약관이 '
 '무효로 된 경우와 회사가 승낙 전에 무 효임을 알았거나 알 수 있었음에도 보험료를 반환하지 않은 경우에는 보험료를 납입한 날의 다음날부터 '
 '반환일까지의 기간에 대하여 회사는 보험계약대출이율을 연단위 복리로 계산한 금액을 더하여 돌려 드립니다.\n'
 '제16조(타인을 위한 계약)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 184},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000626',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
