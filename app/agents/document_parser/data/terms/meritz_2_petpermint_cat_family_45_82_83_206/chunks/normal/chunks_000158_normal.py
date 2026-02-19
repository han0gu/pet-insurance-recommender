from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항의 중도인출금을 지급받은 경우에는「보험료 및 해 약환급금 산출방법서」에 따라 계약자적립액에서 해당 중도 인출금을 '
 '차감합니다.\n'
 '【보험연도】\n'
 '당해 연도 보험계약 해당일부터 차년도 보험계약 해당일 전일까지 매1년 단위의 연도를 말합니다. 예를 들어, 보 험계약일이 2023년 4월 '
 '1일인 경우 보험연도는 4월 1일 부터 차년도 3월 31일까지 1년을 말합니다.\n'
 '【중도인출금의 한도 예시】'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 78},
 'term_type': 'basic',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000158',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
