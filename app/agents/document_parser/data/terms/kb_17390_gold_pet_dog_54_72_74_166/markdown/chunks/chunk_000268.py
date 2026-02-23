from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- |\n'
 '# 5. 선천적 기형 및 이에 근거한 병상제4조(수술의 정의와 장소)\n'
 '\uf000 이 특별약관에 있어서 "수술"이라 함은 병원 또는 의원의 의사, 치과의사 면허를\n'
 '가진 자(이하 "의사"라 합니다)에 의하여 치료가 필요하다고 인정한 경우로서 자\n'
 '택 등에서 치료가 곤란하여 의료기관에서 의사의 관리 하에 직접적인 치료를 목\n'
 '적으로 의료기구를 사용하여 생체(生體)에 절단(切斷), 절제(切除) 등의 조작(操\n'
 '作)을 가하는 것을 말합니다.\n'
 '\uf000 제1항의 수술에서 보건복지부 산하 신의료기술평가위원회(향후 제도 변경 시에는'),
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
