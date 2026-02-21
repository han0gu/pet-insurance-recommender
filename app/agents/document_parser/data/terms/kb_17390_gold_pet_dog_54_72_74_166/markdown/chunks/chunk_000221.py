from langchain_core.documents import Document

chunk = Document(
    page_content=('약특별관KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 71- 71 -제2절 보통약관의 보장일반상해80%이상후유장해제1조(보험금의 '
 '지급사유)\n'
 '회사는 피보험자가 이 보장의 보험기간 중에 상해로 장해분류표(【별표1】(장해분류\n'
 '표) 참조. 이하 같습니다)에서 정한 80%이상 장해지급률에 해당하는 장해상태가 되\n'
 '었을 때에는 최초 1회에 한하여 이 보장의 보험가입금액 전액을 일반상해80%이상후# 유장해보험금으로 보험수익자에게 지급합니다.- '
 '제2조(보험금 지급에 관한 세부규정)'),
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
