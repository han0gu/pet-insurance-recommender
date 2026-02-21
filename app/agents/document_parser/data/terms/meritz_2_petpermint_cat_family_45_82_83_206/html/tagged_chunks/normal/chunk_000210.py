from langchain_core.documents import Document

chunk = Document(
    page_content=('. 강제집행이란 국가의 집행기관이 채권자를 위하여 집<br>행권원에 표시된 사법상의 청구권을 국가권력으로 강<br>제적으로 실현시키는 '
 '것을 말합니다.<br>2. 담보권실행이란 담보권을 설정한 채권자가 채무를 이<br>행하지 않은 채무자에 대하여 해당 담보권을 '
 '실행하<br>는 것을 말합니다.<br>3'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000210',
              'chunk_char_len': 167,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
