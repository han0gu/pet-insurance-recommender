from langchain_core.documents import Document

chunk = Document(
    page_content=('AEA004 | 기타 근골격 계통의 양성 신생물(뒷다리)\n'
 'AEB004 | 기타 근골격 계통의 악성 신생물(뒷다리)\n'
 'AEC004 | 기타 근골격 계통의 악성 신생물(뒷다리) (양 성 또는 악성이 불확실한)\n'
 'NAA001 | 고관절 이형성증 (좌측)\n'
 'NAA002 | 고관절 이형성증 (우측)\n'
 'NAA003 | 고관절 (아) 탈구 (좌측)\n'
 'NAA004 | 고관절 (아) 탈구 (우측)\n'
 'NAA005 | 무혈성골두괴사(LCPD) (좌측)\n'
 'NAA006 | 무혈성골두괴사(LCPD) (우측)\n'
 'NAA007 | 슬개골 (아) 탈구- (좌측-1기)'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 169},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000588',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
