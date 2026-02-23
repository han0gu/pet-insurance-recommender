from langchain_core.documents import Document

chunk = Document(
    page_content=('| (2)요척골 동시, 경비골 동시 | N0974 |\n'
 '| 라. 쇄골, 슬개골, 수근골, 족근골 | N0975 |\n'
 '| 마. 중수골, 중족골, 지골 | N0976 |\n'
 '|  | 별표8 부목치료 대상 분류표 | 별표8 부목치료 대상 분류표 |\n'
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
