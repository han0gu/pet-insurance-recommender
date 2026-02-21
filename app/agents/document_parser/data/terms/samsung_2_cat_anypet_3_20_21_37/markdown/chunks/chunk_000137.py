from langchain_core.documents import Document

chunk = Document(
    page_content=('초과하는 경우 그 초과하는 금액은 각각 없는 것으로 한다.- 1. 기본공제대상자 중 장애인을 피보험자 또는 수익자로 하는 '
 '장애인전용보험으로서 대통령령으로 정\n'
 '- 하는 장애인전용보장성보험료\n'
 '- 2. 기본공제대상자를 피보험자로 하는 대통령령으로 정하는 보험료(제1호에 따른 장애인전용보장성보\n'
 '- 험료는 제외한다)\n'
 '# <소득세법 시행령 제118조의4 (보험료의 세액공제)>- ① 소득세법 제59조의4 제1항 제1호에서 "대통령령으로 정하는 '
 '장애인전용보장성보험료"란 제2항 각'),
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
