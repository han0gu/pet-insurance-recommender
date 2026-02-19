from langchain_core.documents import Document

chunk = Document(
    page_content=('<지급보험금의 계산>\n'
 '{(피보험자가 부담한 1일당 의료비 – 의료비 자기부담금 – 반려견의료비(치과및구강질환포함)(수술\n'
 '당일제외,검사비포함)(재가입형) 보험금의 1일한도) × 보상비율}과 보험증권에 기재된 1회당\n'
 '보상한도액 중 적은 금액\n'
 '단, 반려견의료비보험금이 3-1. 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재 가입형) 특별약관 1일당 보상한도액보다 '
 '적을 경우 보험금을 지급하지 않습니다.\n'
 '<예시안내>'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 77},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['dental']},
 'indexing': {'chunk_id': 'chunk_000461',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
