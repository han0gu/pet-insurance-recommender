from langchain_core.documents import Document

chunk = Document(
    page_content=('제18조(사기에 의한 계약)\n'
 '\uf000 계약자 또는 피보험자가 대리진단, 약물사용을 수단으로 진단절차를 통과하거나 진단서 위·변조 또는 청약일 이전 에 암 또는 '
 '인간면역결핍바이러스(HIV) 감염의 진단 확정을 받은 후 이를 숨기고 가입하는 등 사기에 의하여 계약이 성 립되었음을 회사가 증명하는 '
 '경우에는 계약일부터 5년 이내 (사기사실을 안 날부터 1개월 이내)에 계약을 취소할 수 있 습니다. \uf000 제1항에 따라 계약이 '
 '취소된 경우에는 회사는 이미 납입 한 보험료를 계약자에게 돌려 드립니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 62},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000074',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
