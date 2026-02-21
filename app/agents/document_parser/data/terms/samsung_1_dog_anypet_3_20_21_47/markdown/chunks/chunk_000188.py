from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 수술비용 확대보장 특별약관 제1조(보상하는 손해) 제2항\n'
 '- 3. 피부병 보장 특별약관 제1조(보상하는 손해) 제2항\n'
 '- 4. 반려동물 사망위로금 특별약관 제2조(보상하지 않는 손해) 제2호\n'
 '【펫샵】 동물보호법 시행규칙에 따라 동물을 분양하는 영업활동을 할 수 있는 영업자를 말합니다.\n'
 '【분양】 펫샵에 유상의 재화를 제공하고 반려동물을 입양하는 행위를 말합니다.② 제1항에도 불구하고, 암, 백내장, 녹내장, 심장질환, '
 '신장질환, 방광질환 및 각종결석이 대기기간'),
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
