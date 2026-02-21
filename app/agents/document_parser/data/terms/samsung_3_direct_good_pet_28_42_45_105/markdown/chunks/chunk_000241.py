from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 그 사실을 안 날부터 1개월 이내에 계약을 해지할 수 있습니다. 다만, 이 경우에도 회사는\n'
 '실제 발생한 보험금 지급사유에 대해서는 보험금을 지급합니다.② 회사가 제1항에 따라 계약을 해지한 경우 회사는 그 취지를 계약자에게 '
 '통지하고 제\n'
 '35조(해약환급금) 제1항에 따른 해약환급금을 지급합니다. 다만, 제1항 제1호에서 보\n'
 '험수익자가 보험금의 일부 보험수익자인 경우에는 지급하지 않은 보험금에 해당하는'),
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
