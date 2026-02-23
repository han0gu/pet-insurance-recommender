from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 증가된 위험과 관계없이 발생한 보험<br>금 지급사유에 관해서는 원래대로 지급합니다.<br>\uf000 계약자 또는 피보험자가 '
 '고의 또는 중대한 과실로 제1항 각 호의 변경사실을 회사<br>에 알리지 않았을 경우 변경후 요율이 변경전 요율보다 높을 때에는 회사는 '
 "그 변<br>경사실을 안 날부터 1개월 이내에 계약자 또는 피보험자에게 제4항에 따라 보장</p><br><p id='7' "
 "data-category='paragraph' style='font-size:14px'>104 KB 금쪽같은"),
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
