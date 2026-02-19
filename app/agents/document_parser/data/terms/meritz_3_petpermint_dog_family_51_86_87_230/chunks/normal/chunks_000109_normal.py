from langchain_core.documents import Document

chunk = Document(
    page_content=('【 감액 】\n'
 '보험료, 보험금, 계약자적립액 등을 산정하는 기준이 되 는 가입금액을 계약시 선택한 금액보다 적은 금액으로 줄이는 것을 말합니다.(이에 '
 '따라 보험료, 보험금 및 해 약환급금도 줄어듭니다)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 73},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000109',
              'chunk_char_len': 111,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
