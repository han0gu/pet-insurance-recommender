from langchain_core.documents import Document

chunk = Document(
    page_content=('를 입은 사람(법인인 경우에는 그 이사 또는 법인의 업무를 집행하는 그 밖의 기관)을 말하 며, 보험증권에 기재된 반려동물의 소유자에 '
 '한합니다.</td></tr><tr><td>반려동물</td><td>보험증권에 기재된 반려동물을 말하며, 이 계 약에서 가입 가능한 '
 '반려동물은 대한민국 내에 서 피보험자와 거주를 함께하고 있는 고양이 (猫)를 말합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000270',
              'chunk_char_len': 194,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
