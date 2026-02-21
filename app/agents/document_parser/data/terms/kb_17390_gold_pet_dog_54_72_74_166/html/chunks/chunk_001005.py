from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 사고일<br>또는 발병일부터 365일 이내의 치료인 경우에 한합니다.<br>\uf000 제1항의 경우 '
 '반려동물주요치료보장개시일은 계약일로부터 그날을 포함하여 30일<br>이 지난날의 다음날로 합니다. 단, 상해(상해로 인한 창상 또는 '
 '교상, 이물섭취<br>를 포함)를 직접적인 원인으로 치료를 받은 경우에는 보험계약일을 보장개시일로<br>합니다'),
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
