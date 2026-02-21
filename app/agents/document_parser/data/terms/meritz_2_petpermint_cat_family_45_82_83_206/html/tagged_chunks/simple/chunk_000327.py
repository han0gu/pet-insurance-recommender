from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>주의의무의 위반이 현저한 과실, 즉 현저한 부주의, 태<br>만의 경우로서 조금만 주의를 "
 "하였다면 충분히 피해의<br>발생을 막을 수 있었음에도 그 주의조차 태만히 한 높은<br>강도의 주의의무위반</p><h1 id='70' "
 "style='font-size:20px'>제9조(알릴 의무 위반의 효과)</h1><br><p id='71' "
 "data-category='paragraph' style='font-size:20px'>\uf000 회사는 아래와 같은 사실이 있을 "
 '경우에는 손해의 발생<br>여부에'),
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
 'indexing': {'chunk_id': 'chunk_000327',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
