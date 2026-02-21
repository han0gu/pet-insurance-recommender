from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한도로 합니다.(자기부담급은 수술 당일 의료비에서 차감합니다.)\n'
 '<지급보험금의 계산>{(피보험자가 부담한 수술 당일 의료비 – 1일당 자기부담금) × 보상비율}과 보험증권에 기재된 1\n'
 '일당 보상한도액 중 적은 금액<예시안내>[반려견 의료비(치과및구강질환포함)(수술당일)(재가입형) 계산]- ∙ 보험가입금액 : 200만원, '
 '보상비율 : 70%, 자기부담금 : 3만원\n'
 '- ∙ 예시1\n'
 '- - 피보험자가 부담한 수술당일 의료비 203만원\n'
 '- - 보험금 지급금액\n'
 '= [(203만원 - 3만원) × 70%, 200만원] 중 적은 금액'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
