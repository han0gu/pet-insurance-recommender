from langchain_core.documents import Document

chunk = Document(
    page_content=('- 특\n'
 '- 기관을 말한다.\n'
 '- 약\n'
 '반려동KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 115- 115 -# \uf000 제1항 제4호의 사고증명서는 수의사법 '
 '제12조(진단서 등)에서 규정한 내용에 따라| 국내의 | 동물병원에서 수의사에 의해 발급한 것이어야 합니다. |\n'
 '| --- | --- |'),
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
