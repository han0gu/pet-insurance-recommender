from langchain_core.documents import Document

chunk = Document(
    page_content=('- 억원을 초과하는 경우는 제외한다.\n'
 '# <소득세법 시행규칙 제61조의3 (공제대상보험료의 범위)>영 제118조의4제2항 각 호 외의 부분에서 "기획재정부령으로 정하는 '
 '것"이란 만기에 환급되는 금액이\n'
 '납입보험료를 초과하지 아니하는 보험으로서 보험계약 또는 보험료납입영수증에 보험료 공제대상임이\n'
 '표시된 보험의 보험료를 말한다.2. 모든 피보험자 또는 모든 보험수익자가 「소득세법 시행령 제107조(장애인의 범위) 제 1항」 에'),
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
