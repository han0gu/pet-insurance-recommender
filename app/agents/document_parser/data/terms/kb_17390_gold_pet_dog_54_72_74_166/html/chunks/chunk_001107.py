from langchain_core.documents import Document

chunk = Document(
    page_content=('않습니다.<br>\uf000 제1항의 반려동물장례비용지원금은 총 장례비용에 보험증권에 기재된 보상비율을<br>곱한 금액이며 보험증권에 '
 '기재된 보험가입금액을 한도로 합니다.<br>\uf000 제1항의 경우 반려동물장례비용지원금보장개시일은 계약일로부터 그날을 포함하<br>여 '
 '30일이 지난날의 다음날로 합니다'),
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
