from langchain_core.documents import Document

chunk = Document(
    page_content=('항목 | 자기 부담금 | 지급 한도\n'
 '통 원 의 료 비 Ⅲ | 통원 중 수술을 하지 않은 날의 경우 | MRI,CT 및 내시경처치 를 받은 날의 경우 | 연간 첫번째 | '
 '1일당 3만원/ 5만원 중 보험증 권에 기재된 자기부 담금 | 1일당 30만원\n'
 '연간 두번째 이상 | 1일당 10만원\n'
 'MRI,CT 및 내시경처치를 받지 않은 날의 경우 | 1일당 10만원\n'
 '통원 중 수술을 한 날의 경우 | 수술당일에 한하여 1일당 200만원'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 147},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000489',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
