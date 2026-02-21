from langchain_core.documents import Document

chunk = Document(
    page_content=('지급금액<br>= [(410만원-3만원)×70%, 250만원] 중 적은금액<br>= 250만원(MRI,CT 및 내시경처치와 수술을 '
 "동시에<br>하더라도 수술한도로 지급)</p><br><p id='13' data-category='paragraph' "
 "style='font-size:20px'>\uf000 수술과 MRI,CT 및 내시경처치를 동일한 날에 시행한 경</p><footer "
 "id='14' style='font-size:14px'>157</footer><p id='15' "
 "data-category='paragraph'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000776',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
