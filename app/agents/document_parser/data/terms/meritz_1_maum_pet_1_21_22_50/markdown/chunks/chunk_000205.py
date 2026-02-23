from langchain_core.documents import Document

chunk = Document(
    page_content=('- 5.「군인공제회법」,「한국교직원공제회법」,「대한지방행정공제회법」,「경찰공제회\n'
 '- 법」및「대한소방공제회법」에 따른 공제\n'
 '- 6. 주택 임차보증금의 반환을 보증하는 것을 목적으로 하는 보험·보증. 다만, 보증대\n'
 '- 상 임차보증금이 3억원을 초과하는 경우는 제외한다.\n'
 '# 【소득세법 시행규칙 제61조의3 (공제대상보험료의 범위)】영 제118조의4 제2항 각 호 외의 부분에서 "기획재정부령으로 정하는 '
 '것"이란 만기에- 44 -환급되는 금액이 납입보험료를 초과하지 아니하는 보험으로서 보험계약 또는 보험료납'),
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
