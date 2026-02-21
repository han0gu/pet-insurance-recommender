from langchain_core.documents import Document

chunk = Document(
    page_content=('로 수의사 등 전문가가 권고하는 일정에 따라 예방접종을 하지 않은 때에는 보상하지 않습니다.- 1. 파보바이러스 감염증\n'
 '- 2. 디스템퍼바이러스 감염증\n'
 '- 3. 코로나바이러스 감염증\n'
 '【펫샵】 동물보호법 시행규칙에 따라 동물을 분양하는 영업활동을 할 수 있는 영업자를 말합니다.\n'
 '【분양】 펫샵에 유상의 재화를 제공하고 반려동물을 입양하는 행위를 말합니다.② 반려동물이 제1항의 사고로 치료를 받던 중에 '
 '보험개시일로부터 30일이 지난 경우에도 보험개시일'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
