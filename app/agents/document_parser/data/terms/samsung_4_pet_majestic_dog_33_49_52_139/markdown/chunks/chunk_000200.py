from langchain_core.documents import Document

chunk = Document(
    page_content=('- 부실한 사항을 알렸다고 인정되는 경우에는 특별약관을 해지할 수 있습니다.\n'
 '③ 제1항에 따라 특별약관을 해지하였을 때에는 제35조(해약환급금)제1항에 따른 해약환# 급금을 계약자에게 지급합니다.- ④ 제1항 '
 '제1호에 의한 특별약관의 해지가 보험금 지급사유 발생 후에 이루어진 경우에\n'
 '- 회사는 보험금을 지급하지 않으며, 보험료 납입면제사유가 발생한 경우 보험료 납입\n'
 '- 을 면제하지 않습니다. 또한, 계약 전 알릴 의무 위반 사실(특별약관 해지 등의 원인이'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
