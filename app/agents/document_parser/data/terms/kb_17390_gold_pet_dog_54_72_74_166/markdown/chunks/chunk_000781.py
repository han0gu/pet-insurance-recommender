from langchain_core.documents import Document

chunk = Document(
    page_content=('- 함)에는 갱신일 현재의 약관 등으로 갱신됩니다. 다만, 계약자는 갱신일 현재의\n'
 '- 약관 등에 대해 90일 이내에 그 계약을 취소할 수 있으며, 이 경우 회사는 계약자\n'
 '- 에게 갱신일 이후 납입한 보장특약의 보험료를 돌려드립니다.\n'
 '|  |\n'
 '| --- |'),
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
