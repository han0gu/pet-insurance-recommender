from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한도에서 공제한 후의 잔액을 기준으로 합니다.\n'
 '- ② 제1항에도 불구하고 계약일부터 10년 이내에 인출하는 경우, 각 인출시점까지의 인출\n'
 '- 금액 총합계는 이미 납입한 보험료를 초과할 수 없습니다.\n'
 '# <용어풀이># [보험년도]당해연도 보험계약 해당일부터 다음년도 보험계약 해당일 전일까지로 매1년 단위의 연도임. 예를\n'
 '들어, 보험계약일이 2022년 4월 1일인 경우 보험년도는 4월 1일부터 2023년도 3월 31일까지 1년\n'
 '을 말함<예시안내>[중도인출금의 한도 예시]'),
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
