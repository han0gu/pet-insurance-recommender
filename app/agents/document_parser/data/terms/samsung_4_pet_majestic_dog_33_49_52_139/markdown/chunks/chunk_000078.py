from langchain_core.documents import Document

chunk = Document(
    page_content=('- 우를 말합니다.\n'
 '- ⑧ 제31조(보험료의 납입을 연체하여 해지된 계약의 부활(효력회복))에서 정한 계약의 부\n'
 '- 활이 이루어진 경우 부활을 청약한 날을 제5항의 청약일로 하여 적용합니다.\n'
 '# 제21조 (청약의 철회)① 계약자는 보험증권을 받은 날 부터 15일 이내에 그 청약을 철회할 수 있습니다. 다만,\n'
 '회사가 건강상태 진단을 지원하는 계약, 보험기간이 90일 이내인 계약 또는 전문금융'),
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
