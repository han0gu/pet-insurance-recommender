from langchain_core.documents import Document

chunk = Document(
    page_content=('을 안 날부터 1개월 이내에 계약자 또는 피보험자에게 제4항에 따라 보장됨을 통보하\n'
 '고 이에 따라 보험금을 지급합니다.# 제13조 (알릴 의무 위반의 효과)① 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생여부에 '
 '관계없이 이 특별약관\n'
 '을 해지할 수 있습니다.- 1. 계약자 또는 피보험자가 고의 또는 중대한 과실로 제11조(계약 전 알릴 의무)를 위\n'
 '- 반하고 그 의무가 중요한 사항에 해당하는 경우\n'
 '- 2. 뚜렷한 위험의 증가와 관련된 제12조(계약 후 알릴 의무) 제1항에서 정한 계약 후'),
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
