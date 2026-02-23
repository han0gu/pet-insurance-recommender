from langchain_core.documents import Document

chunk = Document(
    page_content=('- 서 자택 등에서 치료가 곤란하여 의료법 제3조(의료기관)에서 규정한 병원, 의원 또는\n'
 '- 국외의 의료관련법에서 정한 의료기관에서 의사의 관리 하에 치료를 직접적인 목적으\n'
 '- 로 기구를 사용하여 생체(生體)에 절단(切断, 특정부위를 잘라 내는 것), 절제(切除, 특\n'
 '- 정부위를 잘라 없애는 것) 등의 조작을 가하는 것을 말합니다.\n'
 '- ② 제1항의 수술은 보건복지부 산하 신의료기술평가위원회(향후 제도변경 시에는 동 위\n'
 '- 원회와 동일한 기능을 수행하는 기관)로부터 안전성과 치료효과를 인정받은 최신 수'),
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
