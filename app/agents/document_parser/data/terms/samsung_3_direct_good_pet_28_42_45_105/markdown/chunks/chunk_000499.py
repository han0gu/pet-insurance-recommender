from langchain_core.documents import Document

chunk = Document(
    page_content=('- ③ 피보험자가 피해자로부터 손해배상의 청구를 받았을 경우에 회사가 필요하다고 인정\n'
 '- 할 때에는 피보험자를 대신하여 회사의 비용으로 이를 해결할 수 있습니다. 이 경우\n'
 '- 회사의 요구가 있으면 계약자 또는 피보험자는 이에 협력하여야 합니다.\n'
 '- ④ 계약자 및 피보험자가 정당한 이유 없이 제2항 및 제3항의 요구에 협조하지 않았을\n'
 '- 때에는 회사는 그로 인하여 늘어난 손해는 보상하지 않습니다.\n'
 '# 제10조 (보험금의 분담)① 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합'),
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
