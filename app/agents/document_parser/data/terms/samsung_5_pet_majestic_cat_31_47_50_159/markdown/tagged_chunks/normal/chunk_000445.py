from langchain_core.documents import Document

chunk = Document(
    page_content=('- 묘는 대한민국 내에서 피보험자와 거주를 함께하고 있는 고양이(猫)를 말합니다.\n'
 '- 다만 아래에 기재된 고양이(猫)는 이 보험의 가입 대상이 아닙니다.\n'
 '- 가. 보험가입 당시의 연령이 생후 60일 이하 또는 만 10세를 초과하는 고양이(猫)\n'
 '- 나. 판매점, 브리더 등이 매매(賣買)를 목적으로 사육 · 관리하는 고양이(猫)\n'
 '- 다. 흥행을 목적으로 사육·관리하는 고양이(猫)\n'
 '- 라. 유기동물 보호센터 등에서 사육·관리하는 고양이(猫)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000445',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
