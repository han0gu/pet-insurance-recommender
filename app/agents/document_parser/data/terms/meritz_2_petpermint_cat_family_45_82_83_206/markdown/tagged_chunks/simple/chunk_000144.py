from langchain_core.documents import Document

chunk = Document(
    page_content=('| 반려동물 | 보험증권에 기재된 반려동물을 말하며, 이 계 약에서 가입 가능한 반려동물은 대한민국 내에 서 피보험자와 거주를 함께하고 '
 '있는 고양이 (猫)를 말합니다. 다만 아래에 기재된 고양이 (猫)는 이 보험의 가입 대상이 아닙니다. 1. 보험가입 당시의 연령이 생후 '
 '60일 이하 또는 만 8세(단, 실속형의 경우 만 10세) 를 초과하는 고양이(猫) 2. 판매점, 브리더 등이 매매(賣買)를 목적으 로 '
 '사육ㆍ관리하는 고양이(猫) 3. 특수한 목적의 고양이(猫) 4. 흥행을 목적으로 사육ㆍ관리하는 고양이 (猫) 5'),
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
 'indexing': {'chunk_id': 'chunk_000144',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
