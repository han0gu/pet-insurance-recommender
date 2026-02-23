from langchain_core.documents import Document

chunk = Document(
    page_content=('근골격계 질환</td><td>AEB003</td><td>뒷다리의 골육종</td></tr><tr><td>AEA004</td><td>기타 '
 '근골격 계통의 양성 신생물(뒷다리)</td></tr><tr><td>AEB004</td><td>기타 근골격 계통의 악성 '
 '신생물(뒷다리)</td></tr><tr><td>AEC004</td><td>기타 근골격 계통의 악성 신생물(뒷다리) (양 성 또는 악성이 '
 '불확실한)</td></tr><tr><td>NAA001</td><td>고관절 이형성증'),
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
 'indexing': {'chunk_id': 'chunk_000847',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
