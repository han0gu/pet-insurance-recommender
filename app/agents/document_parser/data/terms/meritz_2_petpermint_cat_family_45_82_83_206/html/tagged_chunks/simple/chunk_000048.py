from langchain_core.documents import Document

chunk = Document(
    page_content=('지급기한 내에 지급되지 못할 것으로 판단되는 경우 회<br>사가 예상되는 보험금의 일부를 먼저 지급하는 제도로 피보험자<br>가 필요로 '
 "하는 비용을 보전해 주기 위해 회사가 먼저 지급하는<br>임시 교부금을 말합니다.</p><br><p id='67' "
 "data-category='paragraph' style='font-size:16px'>\uf000 회사는 제1항의 규정에 정한 지급기일 "
 '내에 보험금을 지<br>급하지 않았을 때(제2항의 규정에서 정한 지급예정일을 통<br>지한 경우를 포함합니다)에는 그 다음날부터 '
 '지급일까지의<br>기간에'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000048',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
