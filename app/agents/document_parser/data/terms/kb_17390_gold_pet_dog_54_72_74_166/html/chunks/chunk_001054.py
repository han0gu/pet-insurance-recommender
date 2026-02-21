from langchain_core.documents import Document

chunk = Document(
    page_content=('. 사고증명서(진료비 영수증(치료비 세부내역 포함), 진료기록부(수의사가 작<br>관<br>성한 진료차트), MRI, CT, 방사선 촬영 '
 "등 영상검사를 하는 경우 해당 사진</p><br><p id='26' data-category='list' "
 'style=\'font-size:14px\'>(촬영 날짜 및 시간 필수)<br>- "이물제거(내시경)" 시행한 경우 : 이물제거(내시경) '
 '처치가 명시된 진료비<br>영수증(치료비 세부내역 포함), 내시경영상검사결과지 등<br>- "이물제거(구토유도약물)" 시행한 경우 : '
 '구토유도약물 처방이 명시된'),
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
