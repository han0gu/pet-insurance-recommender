from langchain_core.documents import Document

chunk = Document(
    page_content=('시</td><td>특</td></tr><tr><td>손해배상청구에 ① 손해배상책임 사고발생 별 피보험자 피해자 보험사 관 ③ '
 '대신하여</td><td>대한 회사의 해결 예시 ② 보험금 지급 청구 약 피보험자를</td></tr></tbody></table> '
 '항변으로써 대항가능 ※ 항변이란 어떤 일을 부당하다고 여겨 따지거나 반대하는 뜻을 밝힌다는 '
 "것</td></tr></tbody></table><br><p id='208' data-category='paragraph' "
 "style='font-size:16px'>을 의미합니다.</p><p"),
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
