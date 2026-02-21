from langchain_core.documents import Document

chunk = Document(
    page_content=('- 손해보상의 원인이 생긴 때부터 이 특별약관은 소멸되며 그 때부터 효력이 없습니다.\n'
 '- 이 경우 회사는 이 특별약관의 해약환급금을 지급하지 않습니다.\n'
 '- ② 피보험자가 보험기간 중에 이 특별약관에서 보장하지 않는 사유로 사망하였을 경우에\n'
 '- 는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 회사가 적립한 사망당시\n'
 '- 이 특별약관의 계약자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은\n'
 '- 더 이상 효력이 없습니다.\n'
 '- 62 --'),
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
