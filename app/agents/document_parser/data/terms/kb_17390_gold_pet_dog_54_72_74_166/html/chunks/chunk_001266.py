from langchain_core.documents import Document

chunk = Document(
    page_content=('계산한 지급보험금의 합계액</p><br><figure id=\'89\'><img alt="" '
 'data-coord="top-left:(129,355); bottom-right:(662,459)" /></figure><br><p '
 "id='90' data-category='paragraph' style='font-size:14px'>\uf000 피보험자가 다른 계약에 "
 '대하여 보험금 청구를 포기한 경우에도 회사의 제1항에<br>의한 지급보험금 결정에는 영향을 미치지 않습니다.</p><br><p '
 "id='91'"),
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
