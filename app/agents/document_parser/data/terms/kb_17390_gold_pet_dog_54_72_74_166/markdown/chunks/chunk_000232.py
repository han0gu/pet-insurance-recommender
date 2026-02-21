from langchain_core.documents import Document

chunk = Document(
    page_content=('- 적립액 및 미경과보험료를 계약자에게 지급합니다.\n'
 '제4조(준용규정)이 보장에서 정하지않은 사항은 보통약관 제1절 일반조항을 따릅니다.- 72 -특별약관제1장 상해 관련 특별약관- 73 '
 '-# 제1장 상해 관련 특별약관1. 반려동물양육자금Ⅰ(일반상해사망)제1조(보험금의 지급사유)\n'
 '회사는 이 특별약관의 보험기간 중에 피보험자가 상해의 직접결과로써 사망한 경우\n'
 '에는 이 특별약관의 보험가입금액 전액을 반려동물양육자금Ⅰ(일반상해사망)으로보험수익자에게 지급합니다.제2조(보험금 지급에 관한 세부규정)'),
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
