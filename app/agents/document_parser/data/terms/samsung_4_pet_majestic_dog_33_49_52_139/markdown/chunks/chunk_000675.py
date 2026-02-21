from langchain_core.documents import Document

chunk = Document(
    page_content=('- 합니다) 중에 진단확정된 질병으로 병원 또는 의원(한방병원 또는 한의원을 포함합니\n'
 '- 다)에 1일이상 계속 입원하여 치료를 받은 경우에는 입원기간 동안 보험증권에 기재\n'
 '- 된 반려견을 수탁기관에 위탁함으로써 발생한 위탁비용을 반려견 위탁비용으로 보험\n'
 '- 수익자에게 지급합니다. 다만, 반려견 위탁비용의 지급일수는 1회 입원당 180일을 한\n'
 '- 도로 하며, 피보험자의 입원기간을 초과할 수 없습니다.\n'
 '- ② 제1항의 「수탁기관」 이라 함은 동물보호법 시행규칙 제43조(등록영업의 세부 범위)'),
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
