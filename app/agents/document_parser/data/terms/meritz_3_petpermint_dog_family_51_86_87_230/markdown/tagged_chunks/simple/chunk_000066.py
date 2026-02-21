from langchain_core.documents import Document

chunk = Document(
    page_content=('절통지와 함께 받은 금액을 계약자에게 돌려 드리며, 보험\n'
 '료를 받은 기간에 대하여 평균공시이율 + 1%를 연단위 복리\n'
 '로 계산한 금액을 더하여 지급합니다. 다만, 회사는 계약자\n'
 '가 제1회 보험료를 신용카드로 납입한 계약의 승낙을 거절67하는 경우에는 신용카드의 매출을 취소하며 이자를 더하여\n'
 '지급하지 않습니다.\n'
 '\uf000 회사가 제2항에 따라 일부보장 제외 조건을 붙여 승낙하\n'
 '였더라도 청약일로부터 5년(갱신형 계약의 경우에는 최초\n'
 '계약의 청약일 이후 5년)이 지나는 동안 보장이 제외되는'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000066',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
