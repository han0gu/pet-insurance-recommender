from langchain_core.documents import Document

chunk = Document(
    page_content=("이익을 위해 자기의 이름으로 체결하는 보험계약을 말합니다.</p><h1 id='111' "
 "style='font-size:14px'>제19조(특별약관의 소멸)</h1><br><p id='112' "
 "data-category='paragraph' style='font-size:14px'>이 특별약관의 반려동물의 사망으로 이 특별약관에서 "
 '규정하는 보험금 지급사유가 더 이상<br>발생할 수 없는 경우에는 이 특별약관은 그 때부터 효력이 없습니다.</p><footer '
 "id='113' style='font-size:14px'>- 29"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
