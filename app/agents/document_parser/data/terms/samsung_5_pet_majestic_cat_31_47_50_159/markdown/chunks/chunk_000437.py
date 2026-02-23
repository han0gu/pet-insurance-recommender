from langchain_core.documents import Document

chunk = Document(
    page_content=('# 여 의료법 제3조(의료기관)에서 규정한 국내의 병원 또는 의원에서 의사의 관리 하에\n'
 '[별표-상해및질병관련3]급여 창상봉합술(안면부,단순봉합제외) 대상 수가코드에서 정\n'
 '한 진료행위로 치료를 받은 경우를 말합니다.- ⑤ 제1항 내지 제4항의 창상봉합술은 진료비세부내역서상 보건복지부에서 고시하는 「건\n'
 '- 강보험 행위 급여·비급여 목록 및 급여 상대가치점수」 에서 정한 수가코드 기준을 따\n'
 '- 르며, 이 특별약관 체결 시점 이후 보건복지부에서 고시하는 「건강보험 행위 급여·비'),
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
