from langchain_core.documents import Document

chunk = Document(
    page_content=('계약체결에 대한 회사의 법위반사항이 있는 경우 계약체결일부터 5년 이내의 범위에\n'
 '서 계약자가 위반사항을 안 날부터 1년 이내에 계약해지요구서에 증빙서류를 첨부하# 여 위법계약의 해지를 요구할 수 있습니다.# '
 "<용어풀이># [위법계약]금융상품판매업자등이 '금융소비자보호에 관한 법률' 제47조에서 정한 적합성원칙, 적정성원칙, 설\n"
 '명의무, 불공정영업행위의 금지 또는 부당권유행위 금지를 위반한 계약을 말합니다.- ② 회사는 해지요구를 받은 날부터 10일 이내에 '
 '수락여부를 계약자에 통지하여야 하며,'),
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
