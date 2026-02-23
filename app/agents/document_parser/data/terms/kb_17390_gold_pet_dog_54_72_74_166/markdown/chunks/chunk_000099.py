from langchain_core.documents import Document

chunk = Document(
    page_content=('. ∙ 보험금 삭감 일반적인 경우보다 위험이 높은 피보험자가 가입하기 위한 방법의 하나로, 보 험 가입 후 기간이 경과함에 따라 위험의 '
 '크기 및 정도가 점차 감소하는 위험에 대해 적용하여 보험 가입 후 일정기간 내에 보험사고가 발생할 경우 미리 정해 진 비율로 보험금을 '
 '감액하여 지급하는 방법을 말합니다'),
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
