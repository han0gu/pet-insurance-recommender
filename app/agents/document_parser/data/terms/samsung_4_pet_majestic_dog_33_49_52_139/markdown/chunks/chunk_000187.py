from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나. 직업이 없는 자가 취직한 경우\n'
 '- 다. 현재의 직업을 그만둔 경우\n'
 '<용어풀이>[직업]- 1) 생계유지 등을 위하여 일정한 기간동안(예: 6개월 이상) 계속하여 종사하는 일\n'
 '- 2) 1)에 해당하지 않는 경우에는 개인의 사회적 신분에 따르는 위치나 자리를 말함\n'
 '- 예 ) 학생, 미취학아동, 무직 등\n'
 '[직무]직책이나 직업상 책임을 지고 담당하여 맡은 일- 2. 보험증권 등에 기재된 피보험자의 운전 목적이 변경된 경우\n'
 '- 예) 자가용에서 영업용으로 변경, 영업용에서 자가용으로 변경 등'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
