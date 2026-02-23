from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항에서 의료기관이라 함은 의료법 제3조(의료기관) 제2항에서 정한 국내의 병\n'
 '78 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)| 원이나 의원 또는 국외의 | 의료관련법에서 정한 말합니다. | 의료기관을 |\n'
 '| --- | --- | --- |'),
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
