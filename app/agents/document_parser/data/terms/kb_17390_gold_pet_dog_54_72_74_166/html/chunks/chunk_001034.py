from langchain_core.documents import Document

chunk = Document(
    page_content=('성분 약 물 \uf000 제1항 제6호에서 "특정재활치료Ⅱ"라 함은 수의사가 반려동물의 상해 또는 '
 "질병의</td></tr></tbody></table><br><p id='5' data-category='list' "
 "style='font-size:14px'>치료를 직접적인 목적으로 시행한 저주파자극치료, 초음파 치료, 체외충격파 또<br>는 플라즈마 "
 '물리치료를 말합니다.<br>\uf000 제1항 제7호에서 "항암약물치료"라 함은 수의사가 반려동물의 암의 치료를 직접<br>적인 목적으로 '
 '화학요법 항암제 또는 Tyrosine kinase'),
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
