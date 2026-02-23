from langchain_core.documents import Document

chunk = Document(
    page_content=('에 그 이자를 원금에 더한 금액을 다음 1년의 원금으로\n'
 '하는 이자 계산방법을 말합니다.\n'
 '원금 100원, 이자율 연 10%를 가정할 때- - 1년 후 원리금 : 100원 + (100원×10%) = 110원\n'
 '- - 2년 후 원리금 : 110원 + (110원×10%) = 121원\n'
 '# \uf000 기간과 날짜 관련 용어| 용어 | 정의 |\n'
 '| --- | --- |\n'
 '| 보험기간 | 계약에 따라 보장을 받는 기간을 말합니다. |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000151',
              'chunk_char_len': 230,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
