from langchain_core.documents import Document

chunk = Document(
    page_content=('권리는 소멸됩니다.- 17 -당신에게 좋은보험 삼성화재# 제30조(보험료의 환급)# ① 이 계약이 무효, 효력상실 또는 해지된 때에는 '
 '다음과 같이 보험료를 돌려드립니다.- 1. 계약자 또는 피보험자의 책임 없는 사유에 의하는 경우 : 무효의 경우에는 회사에 납입한 '
 '보험료\n'
 '- 의 전액, 효력상실 또는 해지의 경우에는 경과하지 않은 기간에 대하여 일단위로 계산한 보험료\n'
 '- 2. 계약자 또는 피보험자의 책임 있는 사유에 의하는 경우 : 이미 경과한 기간에 대하여 단기요율'),
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
