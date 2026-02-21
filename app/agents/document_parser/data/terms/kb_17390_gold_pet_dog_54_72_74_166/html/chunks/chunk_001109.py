from langchain_core.documents import Document

chunk = Document(
    page_content=('colspan="2">2024년 4월 10일 2024년 5월 9일</td></tr></tbody></table><br><p '
 "id='100' data-category='paragraph' style='font-size:14px'>\uf000 제3항에도 불구하고 "
 '"제도성 특별약관 - 보장특약 자동갱신(추가납입형) 특별약<br>관"에서 정한 방법에 따라 갱신된 계약의 '
 "반려동물장례비용지원금보장개시일은<br>이 특별약관의 갱신일로 합니다.</p><h1 id='101' "
 "style='font-size:14px'>제2조(보험금 지급에 관한"),
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
