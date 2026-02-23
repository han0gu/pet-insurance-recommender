from langchain_core.documents import Document

chunk = Document(
    page_content=('약의 소멸) 및 제36조(중도인출)는 제외합니다.98 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)- 98 -특별약관제4장 반려동물 '
 '관련 특별약관- 99 -# 제4장 반려동물 관련 특별약관# 반려동물(강아지) 일반조항# 제1조(목적)이 특별약관은 계약자와 회사 사이에 '
 '보험증권에 기재된 반려동물의 상해 또는 질병으로 인한 위험을 보장하기 위하여 체결됩니다.제2조(용어의 정의)'),
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
