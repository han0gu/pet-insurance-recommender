from langchain_core.documents import Document

chunk = Document(
    page_content=('【중도인출금의 한도 예시】\n'
 '계약자가 요청한 시점에서 계산된 기본계약 해약환급금 과 기본계약 적립부분 해약환급금 중 적은 금액이 100만 원인 경우 중도인출 가능액은 '
 '80만원(100만원의 80%)이 며, 보험계약대출금(원금과 이자의 합계가 30만원이라고 가정)이 있는 경우 중도인출 가능액은 '
 '50만원(80만원-30 만원)입니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 78},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000159',
              'chunk_char_len': 182,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
