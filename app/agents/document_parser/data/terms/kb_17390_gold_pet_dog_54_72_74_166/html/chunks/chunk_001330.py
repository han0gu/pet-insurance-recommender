from langchain_core.documents import Document

chunk = Document(
    page_content=('해지된 계약의 부활(효력회복)) 및 제30조(강제집행 등으로 인하여 해지된 계약<br>의 특별부활(효력회복))에 따라 보험계약과 동시에 '
 "이 특별약관의 부활(효력회복)을<br>취급합니다.</p><p id='194' data-category='paragraph' "
 "style='font-size:14px'>제4조(준용규정)</p><br><p id='195' "
 "data-category='paragraph' style='font-size:14px'>이 특별약관에서 정하지 않은 사항은 보험계약을 "
 "따릅니다.</p><p id='196'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
