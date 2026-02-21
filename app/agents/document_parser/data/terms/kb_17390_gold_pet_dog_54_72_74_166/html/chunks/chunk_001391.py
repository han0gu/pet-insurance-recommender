from langchain_core.documents import Document

chunk = Document(
    page_content=(". 만약, 이</p><br><header id='35' style='font-size:14px'>보험료를 "
 "납입하지</header><br><p id='36' data-category='paragraph' "
 "style='font-size:14px'>않으면 회사는 지급할 보험금에서 이를 공제할 수 있습니다.</p><br><p id='37' "
 "data-category='paragraph' style='font-size:14px'>제5조(갱신보장특약의 "
 '보장개시)<br>제2조(보장특약의 자동갱신)의 규정에 따라 계약이 갱신되는 경우,'),
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
