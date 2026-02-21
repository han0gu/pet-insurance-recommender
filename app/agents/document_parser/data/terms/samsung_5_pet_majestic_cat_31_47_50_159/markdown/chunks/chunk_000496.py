from langchain_core.documents import Document

chunk = Document(
    page_content=('는 피보험자가 사실대로 알리는 것을 방해한 경우, 계약자 또는 피보험자에게 사실\n'
 '대로 알리지 않게 하였거나 부실한 사항을 알릴 것을 권유했을 때. 다만, 보험설계\n'
 '사 등의 행위가 없었다 하더라도 계약자 또는 피보험자가 사실대로 알리지 않거나\n'
 '부실한 사항을 알렸다고 인정되는 경우에는 이 특별약관을 해지할 수 있습니다.- ③ 제1항에 따라 이 특별약관을 해지하였을 때에는 이 '
 '특별약관의 해약환급금을 계약자\n'
 '- 에게 지급합니다.\n'
 '- ④ 제1항 제1호에 의한 이 특별약관의 해지가 보험금 지급사유 발생 후에 이루어진 경우'),
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
