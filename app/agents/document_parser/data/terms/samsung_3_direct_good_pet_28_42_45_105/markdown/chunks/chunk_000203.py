from langchain_core.documents import Document

chunk = Document(
    page_content=('- 익자가 변경되었음을 회사에 통지하여야 합니다.\n'
 '<유의사항>계약자가 회사에 보험수익자가 변경되었음을 통지하기 전에 보험금 지급사유가 발생한 경우회사는 변경 전 보험수익자에게 보험금을 '
 '지급할 수 있습니다. 회사가 변경 전 보험수익자에게보험금을 지급한 경우 변경된 보험수익자에게는 별도로 보험금을 지급하지 않습니다.- ③ '
 '회사는 계약자가 제1항에 따라 이 특별약관의 보험가입금액을 감액하고자 할 때에는\n'
 '- 그 감액된 부분은 해지된 것으로 보며, 이로써 회사가 지급하여야 할 해약환급금이 있'),
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
