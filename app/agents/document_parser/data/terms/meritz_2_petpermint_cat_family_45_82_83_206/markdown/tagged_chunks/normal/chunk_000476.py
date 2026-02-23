from langchain_core.documents import Document

chunk = Document(
    page_content=('| 2 | 눈 및 부속 기관의 질환 | FAA007 | 유루증 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA008 | 속눈썹의 질병 (첩모난생 / 첩모중생 / 이소 성첩모) |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FAA009 | 안검내번·외번 |\n'
 '| 2 | 눈 및 부속 기관의 질환 | FBA001 | 궤양성 각막염 · 각막궤양 (각막 미란 포함) |\n'
 '169| 구 분 | 특정질병 | 분류코드 | 항목명 |\n'
 '| --- | --- | --- | --- |\n'
 '|  | 질환 | FBA002 | 각막 이영양증 |'),
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
 'indexing': {'chunk_id': 'chunk_000476',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
