from langchain_core.documents import Document

chunk = Document(
    page_content=('또는 주민등록상에<br>기재된 배우자(이하「배우자」라 합니다)<br>③ 피보험자 본인 또는 배우자와 생계를 같이 하는 동거<br>친족 및 '
 "별거 중인 미혼자녀</p><br><p id='11' data-category='list' "
 "style='font-size:20px'>\uf000 위 제1항에서 피보험자 본인과 본인 이외의 피보험자와<br>의 관계는 사고발생 "
 "당시의 관계를 말합니다.</p><h1 id='12' style='font-size:20px'>제4조(보험금의 청구)</h1><br><p "
 "id='13'"),
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
 'indexing': {'chunk_id': 'chunk_000287',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
