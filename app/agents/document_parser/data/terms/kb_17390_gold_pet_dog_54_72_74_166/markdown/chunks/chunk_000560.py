from langchain_core.documents import Document

chunk = Document(
    page_content=('| ∙ 핵연료물질 : 사용된 | 연료를 포함합니다. |\n'
 '| --- | --- |\n'
 '| ∙ 핵연료물질에 의하여 아래의 치료비, 비용 또는 손해는 보상하지 않습니다. | 오염된 물질 : 원자핵 분열 생성물을 포함합니다. '
 '|\n'
 '# \uf000 회사는1. 반려동물의 선천적, 유전적 질병에 의한 손해(보장개시일 이전부터 객관적으로- 110 -인지할 수 있는 증상을 '
 '포함합니다. 다만 보험기간 중 최초로 발견된 경우에는\n'
 '보상합니다.)2. 질병의 발생일로부터 과거 1년 이내에 예방접종 또는 예방처치를 하지 않아 발| 생한 아래의 | 질병 |'),
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
