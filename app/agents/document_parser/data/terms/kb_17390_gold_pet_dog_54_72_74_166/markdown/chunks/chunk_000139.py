from langchain_core.documents import Document

chunk = Document(
    page_content=('# 우 : 가족관계등록부에 기재된 사망연월일을 기준으로 합니다.| 부 가 설 명 | 실종선고 | 특별 |\n'
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
