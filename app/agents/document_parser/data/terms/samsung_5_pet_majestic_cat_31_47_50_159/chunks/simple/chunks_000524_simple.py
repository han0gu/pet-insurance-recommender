from langchain_core.documents import Document

chunk = Document(
    page_content=('가. 보험가입 당시의 연령이 생후 60일 이하 또는 만 10세를 초과하는 고양이(猫) 나. 판매점, 브리더 등이 매매(賣買)를 목적으로 '
 '사육 · 관리하는 고양이(猫) 다. 흥행을 목적으로 사육·관리하는 고양이(猫) 라. 유기동물 보호센터 등에서 사육·관리하는 고양이(猫)\n'
 '<용어풀이>\n'
 '[흥행]\n'
 '영리를 목적으로 연극, 영화, 서커스 등을 요금을 받고 대중에게 보여주는 행위를 말합니 다.\n'
 '② 지급사유 관련 용어'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000524',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
