from langchain_core.documents import Document

chunk = Document(
    page_content=('나타낸 것을 말합니다.# 제4조 (보험금 지급에 관한 세부규정)- ① 제3조(보험금의 지급사유)에서 장해지급률이 상해 발생일부터 180일 '
 '이내에 확정되지\n'
 '- 않는 경우에는 상해 발생일부터 180일이 되는 날의 의사진단에 기초하여 고정될 것으\n'
 '- 로 인정되는 상태를 장해지급률로 결정합니다. 다만, 장해분류표에 장해판정시기를 별\n'
 '- 도로 정한 경우에는 그에 따릅니다.\n'
 '- ② 제1항에 따라 장해지급률이 결정되었으나 그 이후 보장받을 수 있는 기간(계약의 효'),
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
