from langchain_core.documents import Document

chunk = Document(
    page_content=("id='19' style='font-size:14px'>172</footer><table id='20' "
 "style='font-size:14px'><thead><tr><td>구 "
 '분</td><td>특정질병</td><td>분류코드</td><td>항목명</td></tr></thead><tbody><tr><td '
 'rowspan="26"></td><td rowspan="26"></td><td>KDA016</td><td>소화관 기능 저하 (소화관 정체 '
 '포함)</td></tr><tr><td>KDA017</td><td>항문낭염 / 항문낭'),
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
 'indexing': {'chunk_id': 'chunk_000886',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
