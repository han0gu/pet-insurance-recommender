from langchain_core.documents import Document

chunk = Document(
    page_content=('. | ∙ 상법 제651조(고지의무위반으로 인한 계약해지) 보험계약당시에 보험계약자 또는 피보험자가 고의 또는 중대한 과실로 인하 여 중 '
 '요한 사항을 고지하지 아니하거나 부실의 고지를 한 때에는 보험자는 그 사실을 안 날로부터 1월내에, 계약을 체결한 날로부터 3년내에 '
 '한하여 계약을 해지할 수 있다. 그러나 보험자가 계약당시에 그 사실을 알았거나 중대한 과실로 인하여 알 지 못한 때에는 그러하지 '
 '아니하다. ∙ 상법 제651조의2(서면에 의한 질문의 효력) 보험자가 서면으로 질문한 사항은 중요한 사항으로 추정한다. |'),
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
