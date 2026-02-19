from langchain_core.documents import Document

chunk = Document(
    page_content=('8) 상기 장해항목에 해당되지 않는 장기간의 간병이 필요한 만성질환(만성간질환, 만성폐쇄성폐질환 등)은 장해의 평가 대상으로 인정하지 '
 '않는다.\n'
 '13. 신경계 · 정신행동 장해\n'
 '가. 장해의 분류\n'
 '장 해 의 분 류 | 지급률(%)\n'
 '1) 신경계에 장해가 남아 일상생활 기본동작에 제한을 남긴 때 | 10~100\n'
 '2) 정신행동에 극심한 장해를 남긴때 | 100\n'
 '3) 정신행동에 심한 장해를 남긴 때 | 75\n'
 '4) 정신행동에 뚜렷한 장해를 남긴 때 | 50\n'
 '5) 정신행동에 약간의 장해를 남긴 때 | 25'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 147},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['head', 'other']},
 'indexing': {'chunk_id': 'chunk_000969',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
