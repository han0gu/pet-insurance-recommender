from langchain_core.documents import Document

chunk = Document(
    page_content=('항은 농림축산식품부령으로 정한다.# 제9조 (보험금의 지급절차)① 회사는 제8조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 '
 '드리고 휴대전\n'
 '화 문자메시지 또는 전자우편 등으로 송부하며, 그 서류를 접수한 날부터 3영업일 이\n'
 '내에 보험금을 지급합니다.\n'
 '② 회사가 보험금 지급사유를 조사 · 확인하기 위해 필요한 기간이 제1항의 지급기일을\n'
 '초과할 것이 명백히 예상되는 경우에는 그 구체적 사유와 지급예정일 및 보험금 가지\n'
 '급 제도(회사가 추정하는 보험금의 50% 이내를 지급)에 대하여 피보험자 또는 보험수'),
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
