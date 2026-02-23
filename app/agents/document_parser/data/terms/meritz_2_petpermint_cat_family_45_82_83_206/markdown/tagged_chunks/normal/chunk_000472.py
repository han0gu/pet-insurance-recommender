from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1 | 뒷다리 근골격계 질환 | NAA014 | 골절 (뒷다리) (우측) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA015 | 성장판 골절 (뒷다리) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA016 | 관절염 (뒷다리) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA017 | 관절염 · 퇴행성 관절염 (뒷다리) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA018 | 뼈연골증 (뒷다리) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA019 | 근염 (뒷다리) |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000472',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
