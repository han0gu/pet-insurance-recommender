from langchain_core.documents import Document

chunk = Document(
    page_content=('<2> | · (B) 보험금 2회 + (C) 보험금 3회 + (D) 보험금 3회\n'
 '<용어풀이> [창상봉합술] 창상봉합술이란 상처로 인해 벌어지거나 수술을 위해 벤 조직을 꿰매어 맞추어 주는 것을 말합니 다. [안면부] '
 '안면부란 이마를 포함하여 경부(목)까지의 얼굴 부분을 말합니다.\n'
 '② 제1항의 「연간」 이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지의 기간을 의미합니다.\n'
 '제2조 (보험금 지급에 관한 세부규정)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000508',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
