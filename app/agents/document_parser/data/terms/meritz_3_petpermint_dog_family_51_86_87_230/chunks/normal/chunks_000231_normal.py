from langchain_core.documents import Document

chunk = Document(
    page_content=('료를 받은 기간에 대하여 평균공시이율 + 1%를 연단위 복리 로 계산한 금액을 더하여 지급합니다. 다만, 회사는 계약자 가 제1회 '
 '보험료를 신용카드로 납입한 계약의 승낙을 거절 하는 경우에는 신용카드의 매출을 취소하며 이자를 더하여 지급하지 않습니다.\n'
 '제12조(특별약관의 무효)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 100},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000231',
              'chunk_char_len': 155,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
