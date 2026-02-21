from langchain_core.documents import Document

chunk = Document(
    page_content=('상책임을 부담함으로써 입은 아래의 손해를 이 약관에 따라 보상하여 드립니다.- 1. 피보험자가 피해자에게 지급할 책임을 지는 법률상의 '
 '손해배상금\n'
 '- 2. 계약자 또는 피보험자가 지출한 아래의 비용\n'
 '- 가. 피보험자가 제7조(손해방지의무) 제1항 제1호의 손해의 방지 또는 경감을 위하여 지출한 필\n'
 '- 요 또는 유익하였던 비용\n'
 '- 나. 피보험자가 제7조(손해방지의무) 제1항 제2호의 제3자로부터 손해의 배상을 받을 수 있는\n'
 '- 그 권리를 지키거나 행사하기 위하여 지출한 필요 또는 유익하였던 비용'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
