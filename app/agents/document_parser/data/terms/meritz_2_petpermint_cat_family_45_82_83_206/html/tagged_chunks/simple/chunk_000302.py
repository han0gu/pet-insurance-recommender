from langchain_core.documents import Document

chunk = Document(
    page_content=("id='32' data-category='paragraph' style='font-size:16px'>보험금이 지급기한 내에 지급되지 "
 '못할 것으로 판단되는<br>경우 회사가 예상되는 보험금의 일부를 먼저 지급하는<br>제도로 피보험자가 필요로 하는 비용을 보전해 주기 '
 "위<br>해 회사가 먼저 지급하는 임시 교부금을 말합니다.</p><br><p id='33' data-category='paragraph' "
 "style='font-size:20px'>\uf000 회사는 제1항에서 정한 지급기일내에 보험금을 지급하지</p><footer"),
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
 'indexing': {'chunk_id': 'chunk_000302',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
