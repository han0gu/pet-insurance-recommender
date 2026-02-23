from langchain_core.documents import Document

chunk = Document(
    page_content=('/ 산립종 / 마이봄선종</td></tr><tr><td>FAA005</td><td>체리아이 · 제3안검 '
 '돌출</td></tr><tr><td>FAA006</td><td>비루관폐쇄</td></tr><tr><td>FAA007</td><td>유루증</td></tr><tr><td>FAA008</td><td>속눈썹의 '
 '질병 (첩모난생 / 첩모중생 / 이소 '
 '성첩모)</td></tr><tr><td>FAA009</td><td>안검내번·외번</td></tr><tr><td>FBA001</td><td>궤양성 '
 '각막염 · 각막궤양 (각막 미란'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000854',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
