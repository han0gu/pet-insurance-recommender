from langchain_core.documents import Document

chunk = Document(
    page_content=('- 보상한도액을 한도로 합니다. (자기부담금은 1일당 의료비에서 차감합니다.)\n'
 '# <지급보험금의 계산>{(피보험자가 부담한 1일당 의료비 - 1일당 자기부담금) × 보상비율}과 보험증권에 기재된 1일당\n'
 '보상한도액 중적은 금액# <예시안내>[반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재가입형) 계산]- · 보험가입금액 : '
 '10만원, 보상비율 : 70%, 자기부담금 :3만원\n'
 '- · 예시1\n'
 '- - 피보험자가 부담한 1일당 의료비 13만원 (수술미발생, 의료비 중 검사비 5만원)\n'
 '- - 보험금 지급금액'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000461',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
