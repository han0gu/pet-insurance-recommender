from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 소득세법 등 관련법규가 제·개정 또는 폐지되는 경우 변경된 법령을 따릅니다.\n'
 '- 45 -당신에게 좋은보험 삼성화재# 펫샵 전용 대기기간 보장 특별약관# 제1조(보상하는 손해)① 회사는 아래 각 호에도 불구하고, '
 '펫샵에서 반려동물을 분양받은 후 가입하는 계약에 한하여 보험\n'
 '개시일로부터 30일 이내(이하 "대기기간")에 발생한 손해를 보상하여 드립니다.- 1. 보통약관 제5조(보상하지 않는 손해) 제1항 '
 '제7호\n'
 '- 2. 수술비용 확대보장 특별약관 제1조(보상하는 손해) 제2항'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
