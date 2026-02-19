from langchain_core.documents import Document

chunk = Document(
    page_content=('또한, 국세 및 지방세 체납시 국세청 및 지방자치단체에 의해 채무자의 해약환급금이 압류될 수 있으며, 체납처 분 절차에 따라 회사는 '
 '채권자에게 해약환급금을 지급하 게 됩니다.\n'
 '제20조(계약자의 임의해지)\n'
 '계약자는 계약이 소멸하기 전에는 언제든지 계약을 해지할 수 있으며, 이 경우 회사는 보통약관 제35조(해약환급금) 제1항에 의한 '
 '해약환급금을 계약자에게 지급합니다. 다만, 타인을 위한 계약의 경우에는 계약자는 그 타인의 동의를 얻거나 보험증권을 소지한 경우에 한하여 '
 '계약을 해지할 수 있습니다.\n'
 '제21조(중대사유로 인한 해지)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 103},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000278',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
