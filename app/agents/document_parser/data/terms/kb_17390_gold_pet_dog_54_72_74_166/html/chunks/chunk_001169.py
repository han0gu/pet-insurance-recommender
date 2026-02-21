from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제4조(보상하는 손해의 범위) 제2호의 "가"목, "나"목 또는 "마"목의 비용 :</p><br><p id=\'181\' '
 "data-category='paragraph' style='font-size:14px'>122 KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01)</p><br><p id='182' data-category='paragraph' "
 "style='font-size:14px'>비용의 전액을 보상합니다.</p><br><p id='183' "
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
