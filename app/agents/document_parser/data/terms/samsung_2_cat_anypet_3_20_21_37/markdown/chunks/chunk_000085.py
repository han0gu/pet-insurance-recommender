from langchain_core.documents import Document

chunk = Document(
    page_content=('경우 소멸시효가 완성되어 보험금을 지급받지 못 할 수 있습니다. 다만, 2025년 1월 1일이 토요일 또는 공휴일\n'
 '일 경우 그 다음 첫 영업일에 소멸시효가 완성됩니다.# 제34조(약관의 해석)① 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하여야 '
 '하며 계약자에 따라 다르게 해석하지\n'
 '않습니다.【신의성실의 원칙】 계약관계의 당사자는 권리를 행사하거나 의무를 이행할 때 상대방의 정당한 이익을'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
