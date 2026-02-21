from langchain_core.documents import Document

chunk = Document(
    page_content=('다.④ 회사가 지급할 제1항에서 정한 의료비보험금은 보험증권에 기재된 자기부담금을 차감\n'
 '한 후 보상비율을 곱한 금액이며 보험증권에 기재된 1일당 보상한도액을 한도로 합니\n'
 '다. (자기부담금은 1일당 의료비에서 차감합니다)<지급보험금의 계산>{(피보험자가 부담한 1일당 의료비 - 1일당 자기부담금) × '
 '보상비율}과 보험증권에 기재된 1일당\n'
 '보상한도액 중 적은 금액<예시안내>[반려묘 의료비(치과및구강질환포함)(재가입형) 계산]· 보험가입금액 : 10만원, 보상비율 : 70%, '
 '자기부담금 : 3만원\n'
 '· 예시1-'),
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
