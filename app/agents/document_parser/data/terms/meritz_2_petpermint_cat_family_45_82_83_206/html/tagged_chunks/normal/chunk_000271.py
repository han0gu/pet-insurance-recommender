from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만 아래에 기재된 고양이 (猫)는 이 보험의 가입 대상이 아닙니다. 1. 보험가입 당시의 연령이 생후 60일 이하 또는 만 '
 '8세(단, 실속형의 경우 만 10세) 를 초과하는 고양이(猫) 2. 판매점, 브리더 등이 매매(賣買)를 목적으 로 사육ㆍ관리하는 '
 '고양이(猫) 3. 특수한 목적의 고양이(猫) 4. 흥행을 목적으로 사육ㆍ관리하는 고양이 (猫) 5'),
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
 'indexing': {'chunk_id': 'chunk_000271',
              'chunk_char_len': 198,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
