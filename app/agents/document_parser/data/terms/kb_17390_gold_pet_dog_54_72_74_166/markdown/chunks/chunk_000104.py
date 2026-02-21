from langchain_core.documents import Document

chunk = Document(
    page_content=('전문금융소비자\n'
 '보험계약에 관한 전문성, 자산규모 등에 비추어 보험계약에 따른 위험감수능력\n'
 '이 있는 자로서, 국가, 지방자치단체, 한국은행, 금융회사, 주권상장법인 등을\n'
 '포함하며 "금융소비자보호에 관한 법률" 제2조(정의) 제9호에서 정하는 전문금융# 소비자를 말합니다.# ∙# 일반금융소비자# '
 '전문금융소비자가 아닌 계약자를 말합니다.관 련 법 규 금융소비자보호에 관한 법률\n'
 '제46조(청약의 철회) ① 금융상품판매업자등과 대통령령으로 각각 정하는 보장\n'
 '성 상품, 투자성 상품, 대출성 상품 또는 금융상품자문에 관한 계약의 청약을 한'),
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
