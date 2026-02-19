from langchain_core.documents import Document

chunk = Document(
    page_content=('절통지와 함께 받은 금액을 계약자에게 돌려 드리며, 보험 료를 받은 기간에 대하여 평균공시이율 + 1%를 연단위 복리 로 계산한 금액을 '
 '더하여 지급합니다. 다만, 회사는 계약자 가 제1회 보험료를 신용카드로 납입한 계약의 승낙을 거절 하는 경우에는 신용카드의 매출을 '
 '취소하며 이자를 더하여 지급하지 않습니다.\n'
 '제12조(특별약관의 무효)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000234',
              'chunk_char_len': 187,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
