from langchain_core.documents import Document

chunk = Document(
    page_content=('확인할 수 있습니다.제도성 특별 약관- 129 -5-1. [갱신형] 특별약관의 자동갱신 특별 약관# 제1조 (적용대상)이 특별약관은 '
 '손해의 보상을 내용으로 하는 이 계약의 다른 특별약관 중 [갱신형] 특별\n'
 '약관(이하 「갱신형 계약」 이라 합니다.)에 대하여 적용합니다.# 제2조 (자동갱신에 관한 사항)① 이 특별약관은 다음 각 호의 조건을 '
 '충족하고 계약자가 갱신하기 직전 종전의 갱신형\n'
 '특별약관(이하 「갱신전 계약」 이라 합니다.)의 보험기간이 끝나는 날의 전일까지 계'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
