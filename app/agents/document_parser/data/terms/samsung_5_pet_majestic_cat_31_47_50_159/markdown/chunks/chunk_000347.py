from langchain_core.documents import Document

chunk = Document(
    page_content=('100개 이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 진료과목마다 전속하\n'
 '는 전문의를 둔 병원을 말합니다.# 제3조 (수술의 정의와 장소)① 이 특별약관에서 「수술」 이라 함은 병원 또는 의원의 의사의 면허를 '
 '가진 자(이하 「의사」 라 합니다)에 의하여 5대골절로 치료가 필요하다고 인정된 경우로서, 자택 등에서\n'
 '의 치료가 곤란하여 의료법 제3조(의료기관)에서 규정한 국내의 병원, 의원 또는 국외\n'
 '의 의료관련법에서 정한 의료기관에서 의사의 관리 하에 5대골절의 치료를 직접적인'),
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
