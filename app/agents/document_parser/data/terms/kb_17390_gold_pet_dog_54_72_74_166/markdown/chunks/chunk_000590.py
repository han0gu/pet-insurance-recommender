from langchain_core.documents import Document

chunk = Document(
    page_content=('- 받은 경우\n'
 '- 반\n'
 '- 5. 상해 또는 질병의 직접적인 치료를 목적으로 "특정약물치료Ⅱ"를 받은 경우\n'
 '- 6. 상해 또는 질병의 직접적인 치료를 목적으로 "특정재활치료Ⅱ"를 받은 경우 려동\n'
 '- 7. 암의 직접적인 치료를 목적으로 "항암약물치료"를 받은 경우 물\n'
 '- 제1항 제1호에서 "자기공명영상(MRI)" 및 "컴퓨터단층촬영(CT)"이라 함은 제1조\n'
 '# \uf000(보험금의 지급사유)에서 정한 수의사에 의하여 진단 및 치료가 필요하다고 인정\n'
 '된 경우로서 아래의 의료행위를 말합니다.\n'
 '제'),
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
