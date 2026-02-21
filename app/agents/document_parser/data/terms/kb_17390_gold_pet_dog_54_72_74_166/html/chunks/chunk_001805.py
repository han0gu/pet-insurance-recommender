from langchain_core.documents import Document

chunk = Document(
    page_content=(". 피보험자가 기재 약관에서 질병에 보 개정으로 여부를</td></tr></tbody></table><br><table id='2' "
 "style='font-size:14px'><thead><tr><td>별표17</td><td>반려동물(강아지)</td><td>특정 질병 "
 '분류표</td></tr></thead><tbody><tr><td rowspan="15">코드 801</td><td '
 'rowspan="15">특정 질병 근골격계질환</td><td>세부 '
 '질병명</td></tr><tr><td></td></tr><tr><td>고관절 탈구 /'),
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
