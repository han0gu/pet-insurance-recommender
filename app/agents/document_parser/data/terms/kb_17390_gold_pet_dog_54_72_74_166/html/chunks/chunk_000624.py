from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문<br>의 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회사가 '
 "전액 부담합니<br>다.</p><h1 id='147' style='font-size:14px'>제3조(특별약관의</h1><br><p "
 "id='148' data-category='paragraph' style='font-size:14px'>소멸)</p><br><p "
 "id='149' data-category='list' style='font-size:14px'>\uf000 회사는 제1조(보험금의 "
 '지급사유)에서 정한'),
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
