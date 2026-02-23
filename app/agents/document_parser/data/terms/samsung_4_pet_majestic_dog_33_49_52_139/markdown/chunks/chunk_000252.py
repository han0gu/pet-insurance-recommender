from langchain_core.documents import Document

chunk = Document(
    page_content=('정하는 이율로 계산한 금액을 더하여 납입하여야 합니다. 다만, 금리연동형보험은 각\n'
 '상품별 사업방법서에서 별도로 정한 이율로 계산합니다.\n'
 '② 제1항에 따라 해지계약을 부활(효력회복)하는 경우에는 제15조(계약 전 알릴 의무),\n'
 '제17조(알릴 의무 위반의 효과), 제18조(사기에 의한 계약), 제19조(특별약관의 성립),\n'
 '제26조(제1회 보험료 및 회사의 보장개시)를 준용합니다. 이때 회사는 해지 전 발생한\n'
 '보험금 지급사유를 이유로 부활(효력회복)을 거절하지 않습니다.'),
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
