from langchain_core.documents import Document

chunk = Document(
    page_content=('우에는 우편, 전화, 방문 등의 방법으로 지체없이 회사에 알려야 합니다.1. 보험증권 등에 기재된 직업 또는 직무의 변경- 가. 현재의 '
 '직업 또는 직무가 변경된 경우\n'
 '- 나. 직업이 없는 자가 취직한 경우\n'
 '- 다. 현재의 직업을 그만둔 경우\n'
 '<용어풀이>[직업]\n'
 '1) 생계유지 등을 위하여 일정한 기간동안(예: 6개월 이상) 계속하여 종사하는 일\n'
 '2) 1)에 해당하지 않는 경우에는 개인의 사회적 신분에 따르는 위치나 자리를 말함\n'
 '예 ) 학생, 미취학아동, 무직 등[직무]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
