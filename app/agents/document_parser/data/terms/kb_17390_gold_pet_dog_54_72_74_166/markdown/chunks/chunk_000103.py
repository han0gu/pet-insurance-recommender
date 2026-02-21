from langchain_core.documents import Document

chunk = Document(
    page_content=('제19조(청약의 철회)# \uf000 계약자는보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 있습니다. 다만, 다음 각 호의 '
 '어느 하나에 해당하는 계약은 청약을 철회할 수 없습니다.\n'
 '1. 회사가 건강상태 진단을 지원하는 계약- 2. 보험기간이 90일 이내인 계약\n'
 '- 3. 전문금융소비자가 체결한 계약\n'
 '62 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)# ∙용 어 풀 이\n'
 '전문금융소비자\n'
 '보험계약에 관한 전문성, 자산규모 등에 비추어 보험계약에 따른 위험감수능력'),
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
