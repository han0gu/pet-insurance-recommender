from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의 구성원이 있는 단체\n'
 '② 제1항의 대상단체에 소속된 자로서 동일한 보험계약을 체결한 5인 이상의 피보험자로 피보험단체\n'
 '를 구성하여야 하며, 단체 구성원의 일부만을 대상으로 가입하는 경우에는 다음의 조건을 모두 충\n'
 '족하여야 합니다.- 1. 단체의 내규에 의한 복지제도로서 노사합의에 의하며, 보험료의 일부를 단체 또는 단체의 대표\n'
 '- 자가 부담하여야 합니다.\n'
 '- 2. 제1항 제2호 및 제3호에 해당하는 단체는 내규에 의해 단체의 대표자와 회사가 협정에 의해 체\n'
 '- 결하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
