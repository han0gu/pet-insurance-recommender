from langchain_core.documents import Document

chunk = Document(
    page_content=('. ∙ 담보권실행 담보권실행이란 담보권을 설정한 채권자가 채무를 이행하지 않는 채무자에 대 하여 해당 담보권을 실행하는 것을 말합니다. '
 '∙ 국세 및 지방세 체납처분 절차 국세 및 지방세 체납처분 절차란 국세 또는 지방세를 체납할 경우 국세 기본법 및 지방세법에 의하여 '
 '체납된 세금에 대하여 가산금 징수, 독촉장 발부 및 재산 압류 등의 집행을 하는 것을 말합니다'),
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
