from langchain_core.documents import Document

chunk = Document(
    page_content=('요양병상)을 갖춘 병원, 치과병원, 한방병원(또는 요양병 통 원) 사항 ∙ 의료법 제3조(의료기관) 내지 제3조의3(종합병원)의 규정에 '
 '의한 종합병원 100개 이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 '
 "진료</td></tr></tbody></table><br><p id='111' data-category='paragraph' "
 "style='font-size:16px'>과목마다 전속하는 전문의를 둘 것</p><p id='112' "
 "data-category='paragraph'"),
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
