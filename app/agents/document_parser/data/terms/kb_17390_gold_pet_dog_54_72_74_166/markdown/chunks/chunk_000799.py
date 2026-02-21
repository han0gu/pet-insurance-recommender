from langchain_core.documents import Document

chunk = Document(
    page_content=('- 공제\n'
 '- 5. "군인공제회법", "한국교직원공제회법", "대한지방행정공제회법", "\n'
 '- 경찰공제회법" 및 "대한소방공제회법"에 따른 공제\n'
 '- 6. 주택 임차보증금의 반환을 보증하는 것을 목적으로 하는 보험·보증.\n'
 '경우는 제외한다.# 다만, 보증대상 임차보증금이 3억원을 초과하는∙ 소득세법 시행규칙 제61조의3 (공제대상보험료의 범위)영 '
 '제118조의4제2항 각 호 외의 부분에서 "기획재정부령으로 정하는 것"이란\n'
 '만기에 환급되는 금액이 납입보험료를 초과하지 아니하는 보험으로서 보험계'),
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
