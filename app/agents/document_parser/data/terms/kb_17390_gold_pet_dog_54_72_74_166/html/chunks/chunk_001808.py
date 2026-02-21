from langchain_core.documents import Document

chunk = Document(
    page_content=('신염</td></tr><tr><td>선천성 비뇨기 질환</td></tr><tr><td>수신증</td></tr><tr><td>신부전 '
 '신우신염</td></tr><tr><td>신장 결석</td></tr><tr><td>요도/요관 결석</td></tr><tr><td>요도/요관 '
 '폐색</td></tr><tr><td>요독증</td></tr><tr><td>이소성 요관</td></tr><tr><td>기타 비뇨기계 '
 '질환</td></tr><tr><td>종양 (비뇨기)</td></tr><tr><td></td></tr></tbody></table><p'),
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
