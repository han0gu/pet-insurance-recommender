from langchain_core.documents import Document

chunk = Document(
    page_content=('# 가. 장해의 분류| 장 해 의 분 류 | 지급률(%) |\n'
 '| --- | --- |\n'
 '| 1) 신경계에 장해가 남아 일상생활 기본동작에 제한을 남긴 때 | 10~100 |\n'
 '| 2) 정신행동에 극심한 장해를 남긴때 | 100 |\n'
 '| 3) 정신행동에 심한 장해를 남긴 때 | 75 |\n'
 '| 4) 정신행동에 뚜렷한 장해를 남긴 때 | 50 |\n'
 '| 5) 정신행동에 약간의 장해를 남긴 때 | 25 |\n'
 '| 6) 정신행동에 경미한 장해를 남긴 때 | 10 |\n'
 '| 7) 극심한 치매 : CDR 척도 5점 | 100 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000831',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
