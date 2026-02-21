from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="28"></td><td>GAA004</td><td>외이도염 (원인 '
 '불명)</td></tr><tr><td>GAA006</td><td>외이염</td></tr><tr><td>GBA001</td><td>중이염</td></tr><tr><td>GCA001</td><td>내이염</td></tr><tr><td>LAA001</td><td>농피증 '
 '/ 세균성 피부염</td></tr><tr><td>LAA002</td><td>말라세지아 '
 '피부염</td></tr><tr><td>LAA003</td><td>피부 사상균증 · 곰팡이성'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['skin']},
 'indexing': {'chunk_id': 'chunk_000877',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
