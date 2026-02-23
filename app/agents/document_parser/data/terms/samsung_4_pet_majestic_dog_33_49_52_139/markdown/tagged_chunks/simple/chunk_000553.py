from langchain_core.documents import Document

chunk = Document(
    page_content=('- 질환포함)(수술당일제외, 검사비포함)(재가입형) 특별약관의 보험금이 1일당 보상한도\n'
 '- 액과 동일한 경우에 한하여 보상합니다.\n'
 '<지급보험금의 계산>{(피보험자가 부담한 1일당 의료비 – 의료비 자기부담금 – 반려견의료비(치과및구강질환포함)(수술\n'
 '당일제외,검사비포함)(재가입형) 보험금의 1일한도) × 보상비율}과 보험증권에 기재된 1회당\n'
 '보상한도액 중 적은 금액단 , 반려견의료비보험금이 4-1. 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000553',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
