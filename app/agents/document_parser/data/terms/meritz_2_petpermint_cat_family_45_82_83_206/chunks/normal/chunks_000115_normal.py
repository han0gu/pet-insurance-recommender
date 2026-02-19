from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 피보험자가 사망한 경우에는 이 계약은 소멸되며, 이 경 우 회사는 그 때까지「보험료 및 해약환급금 산출방법서」 에서 정한 '
 '사망 당시 계약자적립액(중도인출이 있는 경우 중도인출 원금과 이자를 차감하고 적립한 금액을 말합니다) 및 미경과보험료를 계약자에게 '
 '지급합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 70},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000115',
              'chunk_char_len': 151,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
