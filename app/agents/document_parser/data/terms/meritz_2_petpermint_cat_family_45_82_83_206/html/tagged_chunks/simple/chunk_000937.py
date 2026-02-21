from langchain_core.documents import Document

chunk = Document(
    page_content=(". 귓바퀴의 결손</h1><br><p id='23' data-category='list' style='font-size:20px'>1) "
 '“귓바퀴의 대부분이 결손된 때”라 함은 귓바퀴의 연<br>골부가 1/2이상 결손된 경우를 말한다.<br>2) 귓바퀴의 연골부가 1/2 '
 "미만 결손이고 청력에 이상이<br>없으면 외모의 추상(추한 모습)장해로만 평가한다.</p><h1 id='24' "
 "style='font-size:20px'>라"),
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
 'indexing': {'chunk_id': 'chunk_000937',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
