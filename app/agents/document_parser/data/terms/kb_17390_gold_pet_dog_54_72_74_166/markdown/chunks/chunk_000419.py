from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항의 "환경성질환"의 진단확정은 의료법 제3조에서 정한 병원 또는 국외의 의\n'
 '- 료관련법에서 정한 의료기관의 의사자격을 가진 자에 의한 진단서에 의합니다. 제\n'
 '- 도\n'
 '- 또한 회사가 "환경성질환"의 조사나 확인을 위하여 필요하다고 인정하는 경우 검\n'
 '- 성\n'
 '- 사결과, 진료기록부의 사본제출을 요청할 수 있습니다.\n'
 '- 특\n'
 '- 약\n'
 '함은 "환경성질환 분류표"【별표15】- 93 -KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 93동물질- 제4조(입원의 정의와 '
 '장소)'),
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
