from langchain_core.documents import Document

chunk = Document(
    page_content=('| 802 | 비뇨기계질환 | 신부전 신우신염 |\n'
 '| 802 | 비뇨기계질환 | 신장 결석 |\n'
 '| 802 | 비뇨기계질환 | 요도/요관 결석 |\n'
 '| 802 | 비뇨기계질환 | 요도/요관 폐색 |\n'
 '| 802 | 비뇨기계질환 | 요독증 |\n'
 '| 802 | 비뇨기계질환 | 이소성 요관 |\n'
 '| 802 | 비뇨기계질환 | 기타 비뇨기계 질환 |\n'
 '| 802 | 비뇨기계질환 | 종양 (비뇨기) |\n'
 '| 802 | 비뇨기계질환 |  |\n'
 '164 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- 164 -|  | 코드 특정 |  |'),
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
