from langchain_core.documents import Document

chunk = Document(
    page_content=('체납할 경우 국세 기본법 및 지방세법에 의하여 체납\n'
 '된 세금에 대하여 가산금 징수, 독촉장 발부 및 재산\n'
 '압류 등의 집행을 하는 것을 말합니다.- \n'
 '법원은 채권자의 신청에 따른 강제집행 및 담보권실행으\n'
 '로 채무자의 해약환급금을 압류할 수 있으며, 법원의 추\n'
 '심명령 또는 전부명령에 따라 회사는 채권자에게 해약환\n'
 '급금을 지급하게 됩니다.\n'
 '또한, 국세 및 지방세 체납시 국세청 및 지방자치단체에106의해 채무자의 해약환급금이 압류될 수 있으며, 체납처\n'
 '분 절차에 따라 회사는 채권자에게 해약환급금을 지급하'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000218',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
