from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항의 "골절철심제거술"은 의료법 제3조(의료기관)에서 정한 국내의 병원 또는\n'
 '- 국외의 의료관련법에서 정한 의료기관의 의사(치과의사 제외) 면허를 가진 자(이\n'
 '- 제\n'
 '- 하 "의사"라 합니다)에 의하여 "골절철심제거술"이 필요하다고 인정한 경우로서\n'
 '- 도\n'
 '- "의사"의 관리하에 의료법 제3조(의료기관) 제2항에서 규정한 국내의 병원 및 의\n'
 '- 성\n'
 '- 원에서 행한 의료행위에 한합니다.\n'
 '- 특\n'
 '- \uf000 제1항에도 불구하고, 보건복지부에서 고시하는「건강보험 행위 급여․비급여 목록 약'),
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
