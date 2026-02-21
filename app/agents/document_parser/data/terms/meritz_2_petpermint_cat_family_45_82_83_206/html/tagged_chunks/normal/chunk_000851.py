from langchain_core.documents import Document

chunk = Document(
    page_content=('· 퇴행성 관절염 (뒷다리)</td></tr><tr><td>NAA018</td><td>뼈연골증 '
 '(뒷다리)</td></tr><tr><td>NAA019</td><td>근염 '
 '(뒷다리)</td></tr><tr><td>NAA020</td><td>염좌 '
 '(뒷다리)</td></tr><tr><td>NAA021</td><td>기타 근골격계 질환 '
 '(뒷다리)</td></tr><tr><td>NAA022</td><td>고관절 이형성증 / (아) '
 '탈구</td></tr><tr><td>NAA023</td><td>무혈성골두괴사(LCPD) 무릎뼈'),
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
 'indexing': {'chunk_id': 'chunk_000851',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
