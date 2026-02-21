from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1 | 뒷다리 근골격계 질환 | NAA019 | 근염 (뒷다리) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA020 | 염좌 (뒷다리) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA021 | 기타 근골격계 질환 (뒷다리) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA022 | 고관절 이형성증 / (아) 탈구 |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA023 | 무혈성골두괴사(LCPD) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA024 | 무릎뼈 탈구 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000557',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
