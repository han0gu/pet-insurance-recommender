from langchain_core.documents import Document

chunk = Document(
    page_content=("말합니다.</p><br><h1 id='67' style='font-size:18px'>【 보험료 할증 】</h1><br><p "
 "id='68' data-category='paragraph' style='font-size:16px'>일반적인 경우보다 위험이 높은 "
 '피보험자가 가입하기 위<br>한 방법의 하나로, 보험 가입 후 기간이 경과함에 따라<br>위험의 크기 및 정도가 점차 증가하는 위험 또는 '
 '기간의<br>경과에 상관없이 일정한 상태를 유지하는 위험에 적용하<br>는 방법으로 위험 정도에 따라 특별보험료를 추가로 부<br>가하는'),
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
 'indexing': {'chunk_id': 'chunk_000118',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
