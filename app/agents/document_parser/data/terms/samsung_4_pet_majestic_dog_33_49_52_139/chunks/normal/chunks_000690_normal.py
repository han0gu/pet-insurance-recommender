from langchain_core.documents import Document

chunk = Document(
    page_content=('<지급보험금의 계산>\n'
 '{(피보험자가 부담한 수술 당일 의료비 – 1일당 자기부담금) × 보상비율}과 보험증권에 기재된 1 일당 보상한도액 중 적은 금액\n'
 '<예시안내>\n'
 '[반려견 의료비(치과및구강질환포함)(수술당일)(재가입형) 계산]\n'
 '∙ 보험가입금액 : 200만원, 보상비율 : 70%, 자기부담금 : 3만원 ∙ 예시1\n'
 '- 피보험자가 부담한 수술당일 의료비 203만원 - 보험금 지급금액\n'
 '= [(203만원 - 3만원) × 70%, 200만원] 중 적은 금액 = 140만원\n'
 '∙ 예시2'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 114},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000690',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
