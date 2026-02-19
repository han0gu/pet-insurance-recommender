from langchain_core.documents import Document

chunk = Document(
    page_content=('법원은 채권자의 신청에 따른 강제집행 및 담보권실행으 로 채무자의 해약환급금을 압류할 수 있으며, 법원의 추 심명령 또는 전부명령에 따라 '
 '회사는 채권자에게 해약환 급금을 지급하게 됩니다. 또한, 국세 및 지방세 체납시 국세청 및 지방자치단체에'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 106},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000273',
              'chunk_char_len': 135,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
