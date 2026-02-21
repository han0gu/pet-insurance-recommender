from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 용 어 풀 이 현저하게 공정을 잃은 합의 사회통념상 일반 보통인이라면 그 같은 일을 하지 않을 정도로 현저하게 공정 | 용 어 풀 이 '
 '현저하게 공정을 잃은 합의 사회통념상 일반 보통인이라면 그 같은 일을 하지 않을 정도로 현저하게 공정 |\n'
 '성을 잃은 것을 말합니다.제49조(개인정보보호)- \uf000 회사는 이 계약과 관련된 개인정보를 이 계약의 체결, 유지, 보험금 지급 '
 '등을 위\n'
 '- 하여"개인정보 보호법","신용정보의 이용 및 보호에 관한 법률" 등 관계 법령에 정'),
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
