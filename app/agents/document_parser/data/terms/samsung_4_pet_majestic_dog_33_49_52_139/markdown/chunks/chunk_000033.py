from langchain_core.documents import Document

chunk = Document(
    page_content=('- 회사의 서면에 의한 조사요청에 동의하여야 합니다. 다만, 정당한 사유없이 이에 동의\n'
 '- 하지 않을 경우 사실확인이 끝날 때까지 회사는 보험금 지급 지연에 따른 이자를 지\n'
 '- 급하지 않습니다.\n'
 '- ⑦ 회사는 제6항의 서면조사에 대한 동의 요청시 조사목적, 사용처 등을 명시하고 설명합\n'
 '- 니다.\n'
 '# 제9조 (공시이율의 적용 및 공시)① 이 보험의 적립부분 순보험료(적립보험료에서 정해진 계약체결비용 및 계약관리비용\n'
 '을 공제한 금액을 말합니다. 이하 같습니다)에 대한 적립이율은 매월 1일 회사가 정한'),
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
