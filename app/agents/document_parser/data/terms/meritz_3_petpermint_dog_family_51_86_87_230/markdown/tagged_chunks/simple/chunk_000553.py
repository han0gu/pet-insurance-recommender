from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1 | 뒷다리 근골격계 질환 | AEB004 | 기타 근골격 계통의 악성 신생물(뒷다리) |\n'
 '| 1 | 뒷다리 근골격계 질환 | AEC004 | 기타 근골격 계통의 악성 신생물(뒷다리) (양 성 또는 악성이 불확실한) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA001 | 고관절 이형성증 (좌측) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA002 | 고관절 이형성증 (우측) |\n'
 '| 1 | 뒷다리 근골격계 질환 | NAA003 | 고관절 (아) 탈구 (좌측) |'),
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
 'indexing': {'chunk_id': 'chunk_000553',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
