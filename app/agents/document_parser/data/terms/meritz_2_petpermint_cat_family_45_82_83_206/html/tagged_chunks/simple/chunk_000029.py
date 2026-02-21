from langchain_core.documents import Document

chunk = Document(
    page_content=('그 신체의 동일 부위에 또다시 제6<br>항에 규정하는 후유장해상태가 발생하였을 경우에는 직전까<br>지의 후유장해에 대한 '
 '후유장해보험금이 지급된 것으로 보<br>고 최종 후유장해 상태에 해당되는 후유장해보험금에서 이<br>를 차감하여 지급합니다.</p><h1 '
 "id='37' style='font-size:20px'>제5조(보험금을 지급하지 않는 사유)</h1><br><p id='38' "
 "data-category='paragraph' style='font-size:20px'>\uf000 회사는 다음 중 어느 한 가지로 "
 '보험금 지급사유가'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000029',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
