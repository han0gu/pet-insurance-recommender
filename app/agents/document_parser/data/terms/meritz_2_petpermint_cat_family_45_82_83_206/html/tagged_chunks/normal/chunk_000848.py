from langchain_core.documents import Document

chunk = Document(
    page_content=('불확실한)</td></tr><tr><td>NAA001</td><td>고관절 이형성증 '
 '(좌측)</td></tr><tr><td>NAA002</td><td>고관절 이형성증 '
 '(우측)</td></tr><tr><td>NAA003</td><td>고관절 (아) 탈구 '
 '(좌측)</td></tr><tr><td>NAA004</td><td>고관절 (아) 탈구 '
 '(우측)</td></tr><tr><td>NAA005</td><td>무혈성골두괴사(LCPD) '
 '(좌측)</td></tr><tr><td>NAA006</td><td>무혈성골두괴사(LCPD)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000848',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
