from langchain_core.documents import Document

chunk = Document(
    page_content=('날에 두 종류 이상의 "반려동물주요치료"를 받은 경우</td></tr><tr><td colspan="2">예 시 최대 보상한도액의 적용 '
 '특 [동일한 날에 MRI/CT 시행 후 항암약물치료 시행시 최대 보상한도액 예시] 별 ·MRI/CT(100만원), 항암약물치료(30만원) '
 '기준 약 예시① 관 ·MRI/CT 및 항암약물치료에 대한 연간 지급한도가 각각 1회 이상 남아있는 경 우 ·최대 보상한도액 = '
 '{MRI/CT 보상한도액(100만원), 항암약물치료 보상한도액 (30만원)} 중 높은 금액 = 100만원 상 예시② 해 ·MRI/CT에'),
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
