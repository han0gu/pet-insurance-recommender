from langchain_core.documents import Document

chunk = Document(
    page_content=('10만원# - 보험금 지급금액= [(153만원 - 3만원 - 10만원) × 70%, 200만원] 중 적은 금액\n'
 '= 98만원∙ 예시2- - 피보험자가 이물질제거(구토유도약물) 시행 당일 부담한 의료비 : 33만원\n'
 '- - 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함) 특별약관 지급금액 : 10만원\n'
 '- - 보험금 지급금액\n'
 '- = [(33만원 - 3만원 - 10만원) × 70%, 20만원] 중 적은 금액\n'
 '- = 14만원\n'
 '# ∙ 예시3- 피보험자가 이물질제거(구토유도약물) 시행 당일 부담한 의료비 : 10만원'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000555',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
