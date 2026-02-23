from langchain_core.documents import Document

chunk = Document(
    page_content=('- · 수술여부 : 수술을 하지 않은 날의 경우\n'
 '· 예시- 피보험자가 부담한 MRI 또는 CT 촬영 당일 의료비 113만원\n'
 '- 4-1. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관에서 지급한 보험금 : 10만원\n'
 '- 보험금 지급금액- = [(113만원 - 10만원 - 3만원) × 70%, 100만원] 중 적은 금액\n'
 '- = 70만원\n'
 '- ⑤ 제4항의 「자기부담금」 이란 보험증권에 기재된 4-1. 반려묘 의료비(치과및구강질환\n'
 '- 포함)(재가입형) 특별약관의 자기부담금을 말합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
