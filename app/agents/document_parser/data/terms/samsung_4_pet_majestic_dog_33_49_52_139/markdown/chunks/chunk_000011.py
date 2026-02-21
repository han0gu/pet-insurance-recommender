from langchain_core.documents import Document

chunk = Document(
    page_content=('나타낸 것을 말합니다.# 제 4조 (보험금 지급에 관한 세부규정)- ① 제3조(보험금의 지급사유) 에서 장해지급률이 상해 발생일부터 '
 '180일 이내에 확정되\n'
 '- 지 않는 경우에는 상해 발생일부터 180일이 되는 날의 의사진단에 기초하여 고정될\n'
 '- 것으로 인정되는 상태를 장해지급률로 결정합니다. 다만, 장해분류표에 장해판정시기\n'
 '- 를 별도로 정한 경우에는 그에 따릅니다.\n'
 '- ② 제1항에 따라 장해지급률이 결정되었으나 그 이후 보장받을 수 있는 기간(계약의 효'),
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
