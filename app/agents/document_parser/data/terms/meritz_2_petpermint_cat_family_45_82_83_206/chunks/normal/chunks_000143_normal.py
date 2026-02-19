from langchain_core.documents import Document

chunk = Document(
    page_content=('. 법원은 채권자의 신청에 따른 강제집행 및 담보권실행으 로 채무자의 해약환급금을 압류할 수 있으며, 법원의 추 심명령 또는 전부명령에 '
 '따라 회사는 채권자에게 해약환 급금을 지급하게 됩니다. 또한, 국세 및 지방세 체납시 국세청 및 지방자치단체에 의해 채무자의 해약환급금이 '
 '압류될 수 있으며, 체납처 분 절차에 따라 회사는 채권자에게 해약환급금을 지급하 게 됩니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 75},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000143',
              'chunk_char_len': 204,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
