from langchain_core.documents import Document

chunk = Document(
    page_content=('NAA015 | 성장판 골절 (뒷다리)\n'
 'NAA016 | 관절염 (뒷다리)\n'
 'NAA017 | 관절염 · 퇴행성 관절염 (뒷다리)\n'
 'NAA018 | 뼈연골증 (뒷다리)\n'
 'NAA019 | 근염 (뒷다리)\n'
 'NAA020 | 염좌 (뒷다리)\n'
 'NAA021 | 기타 근골격계 질환 (뒷다리)\n'
 'NAA022 | 고관절 이형성증 / (아) 탈구\n'
 'NAA023 | 무혈성골두괴사(LCPD) 무릎뼈 탈구\n'
 'NAA024 | 십자 인대 손상 파열 (전방 / 후방)\n'
 'NAA025 NAA026 | 골절 (뒷다리)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 169},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['joint', 'other', 'other', 'other']},
 'indexing': {'chunk_id': 'chunk_000590',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
