from langchain_core.documents import Document

chunk = Document(
    page_content=('. 한편, 세 번째 갱신계약의 특약보험료는 가입 9년후 새롭게 산출한 보험료표를 적용받는데, 우선 피보험자 반려동물의 나이증가(9세 → '
 '12세)로 인한 보험료 의 증가분과 새롭게 산출된 보험료의 인상분이 함께 반영되어 12,500원을 '
 '납입</td></tr></tbody></table> 상 '
 '<table><thead><tr><td>합니다.</td><td></td><td></td><td></td><td></td><td></td><td>해</td></tr></thead><tbody><tr><td>구'),
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
