from langchain_core.documents import Document

chunk = Document(
    page_content=("회사는 제2항에 따라 손해를 배상할 책임을 집니</p><br><table id='172' "
 "style='font-size:16px'><thead></thead><tbody><tr><td></td><td><table><thead></thead><tbody><tr><td>다.</td></tr></tbody></table></td></tr><tr><td "
 'colspan="2">용 어 풀 이 현저하게 공정을 잃은 합의 사회통념상 일반 보통인이라면 그 같은 일을 하지 않을 정도로 현저하게'),
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
