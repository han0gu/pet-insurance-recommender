from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1 | 뒷다리 근골격계 질환 | NAA019 | 근염 (뒷다리) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA020 | 염좌 (뒷다리) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA021 | 기타 근골격계 질환 (뒷다리) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA022 | 고관절 이형성증 / (아) 탈구 |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA023 | 무혈성골두괴사(LCPD) 무릎뼈 탈구 |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA024 | 십자 인대 손상 파열 (전방 / 후방) |'),
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
 'indexing': {'chunk_id': 'chunk_000473',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
