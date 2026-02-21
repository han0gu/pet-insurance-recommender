from langchain_core.documents import Document

chunk = Document(
    page_content=('. 해당연도의 계약해당일이 없는 경우<br>에는 해당 월의 마지막 날을 계약해당일로 합니다.<br>계약일: 2022년 10월 1일 => '
 "계약해당일: 10월 1일<br>계약일: 2024년 2월 29일 => 계약해당일: 2월 말일</p><h1 id='46' "
 "style='font-size:14px'>제15조(제1회 보험료 및 회사의</h1><br><p id='47' "
 "data-category='paragraph' style='font-size:14px'>보장개시)</p><br><p id='48' "
 "data-category='list'"),
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
