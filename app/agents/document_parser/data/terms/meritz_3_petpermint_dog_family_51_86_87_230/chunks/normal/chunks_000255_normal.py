from langchain_core.documents import Document

chunk = Document(
    page_content=('제16조(제1회 보험료 및 회사의 보장개시)\n'
 '\uf000 회사는 계약의 청약을 승낙하고 제1회 보험료를 받은 때 부터 이 약관이 정한 바에 따라 보장을 합니다. 또한, 회사 가 '
 '청약과 함께 제1회 보험료를 받은 후 승낙한 경우에도 제1회 보험료를 받은 때부터 보장이 개시됩니다. 자동이체 또는 신용카드로 납입하는 '
 '경우에는 자동이체신청 또는 신 용카드매출승인에 필요한 정보를 제공한 때를 제1회 보험료 를 받은 때로 하며, 계약자의 책임 있는 사유로 '
 '자동이체 또는 매출승인이 불가능한 경우에는 보험료가 납입되지 않 은 것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 103},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000255',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
